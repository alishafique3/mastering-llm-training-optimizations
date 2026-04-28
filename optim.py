import os
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


#####################################
# Dataset
#####################################
class TextDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt, tokenizer, batch_size=4, max_length=1024,
                      stride=1024, shuffle=True, drop_last=True, num_workers=0):
    dataset = TextDataset(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=num_workers,
        pin_memory=True,  # 🚀 faster host→GPU transfer
        persistent_workers=(num_workers > 0),  # 🚀 keep workers alive across epochs
    )


#####################################
# Training utilities
#####################################
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device, non_blocking=True)   # 🚀 non_blocking w/ pin_memory
    target_batch = target_batch.to(device, non_blocking=True) # 🚀
    outputs = model(input_ids=input_batch)
    logits = outputs.logits
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    for i, (inp, tgt) in enumerate(data_loader):
        if i >= num_batches:
            break
        with torch.no_grad():
            loss = calc_loss_batch(inp, tgt, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context, max_new_tokens=50):
    model.eval()
    gen_model = model._orig_mod if hasattr(model, "_orig_mod") else model  # 🚀 unwrap torch.compile
    inputs = tokenizer(start_context, return_tensors="pt").to(device)
    with torch.no_grad():
        out = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(out[0], skip_special_tokens=True).replace("\n", " "))
    model.train()


def train_model(model, train_loader, val_loader, optimizer, device,
                num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens = [], [], []
    total_tokens, global_step, last_tokens = 0, -1, 0
    cumulative_tokens, cumulative_time = 0.0, 0.0

    use_cuda = device.type == "cuda"
    if use_cuda:
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t_start.record()
    else:
        t0 = time.time()

    for epoch in range(num_epochs):
        model.train()
        for inp, tgt in train_loader:
            optimizer.zero_grad(set_to_none=True)  # 🚀 slightly faster than zeroing
            global_step += 1

            loss = calc_loss_batch(inp, tgt, model, device)
            loss.backward()
            optimizer.step()

            total_tokens += inp.numel()

            if global_step % eval_freq == 0:
                if use_cuda:
                    t_end.record()
                    torch.cuda.synchronize()
                    elapsed = t_start.elapsed_time(t_end) / 1000
                    t_start.record()
                else:
                    elapsed = time.time() - t0
                    t0 = time.time()

                tokens_interval = total_tokens - last_tokens
                last_tokens = total_tokens
                tps = tokens_interval / elapsed if elapsed > 0 else 0

                if global_step:
                    cumulative_tokens += tokens_interval
                    cumulative_time += elapsed
                avg_tps = cumulative_tokens / cumulative_time if cumulative_time > 0 else 0

                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens.append(total_tokens)

                print(f"Ep {epoch+1}, Step {global_step:06d}, "
                      f"Train: {train_loss:.3f}, Val: {val_loss:.3f}, "
                      f"Step tok/sec: {round(tps)}, Avg tok/sec: {round(avg_tps)}")

        generate_and_print_sample(model, tokenizer, device, start_context)

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"\nAllocated memory: {allocated:.4f} GB")
            print(f"Reserved memory: {reserved:.4f} GB\n")

    return train_losses, val_losses, track_tokens


#####################################
# Main
#####################################
def main(settings):
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"PyTorch version: {torch.__version__}")
    print(f"Using {device}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 7:
            torch.set_float32_matmul_precision("high")   # 🚀 TF32 / tensor cores
            print("Uses tensor cores")

    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 🚀 OPTIMIZED (bf16 + SDPA flash attention):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,            # 🚀 bf16 weights/activations
        attn_implementation="sdpa",      # 🚀 PyTorch SDPA (flash path)
        # attn_implementation="flash_attention_2",  # 🚀 if flash-attn installed
    )
    model.to(device)

    # 🚀 A100 80GB: gradient checkpointing NOT needed — plenty of memory
    model.config.use_cache = False          # 🚀 required during training

    model = torch.compile(model)            # 🚀 compile model (slow 1st step, faster after)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        fused=True,                         # 🚀 fused AdamW kernel (CUDA only)
    )

    file_path = "middlemarch.txt"
    url = "https://www.gutenberg.org/cache/epub/145/pg145.txt"
    if not os.path.exists(file_path):
        import requests
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(r.text)
    with open(file_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    context_length = settings["context_length"]

    train_loader = create_dataloader(
        text_data[:split_idx], tokenizer,
        batch_size=settings["batch_size"],
        max_length=context_length, stride=context_length,
        drop_last=True, shuffle=True,
        num_workers=4,   # 🚀 parallel data loading
    )
    val_loader = create_dataloader(
        text_data[split_idx:], tokenizer,
        batch_size=settings["batch_size"],
        max_length=context_length, stride=context_length,
        drop_last=False, shuffle=False,
        num_workers=4,   # 🚀
    )

    return train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, device=device,
        num_epochs=settings["num_epochs"],
        eval_freq=10, eval_iter=1,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    SETTINGS = {
        "learning_rate": 5e-5,
        "num_epochs": 3,
        "batch_size": 32,        # 🚀 A100 80GB can comfortably handle this at ctx=1024
        "context_length": 1024,
        "weight_decay": 0.1,
    }
    main(SETTINGS)
