# conda create -n nlp310 python=3.10 -y
# source /opt/conda/etc/profile.d/conda.sh
# conda activate nlp310
# 확인 : python -V
# pip install -r requirements.txt
# export TOKENIZERS_PARALLELISM=false
# python src/main.py --config configs/exp.yaml

# (tmux 설치)
#apt-get update
#apt-get install -y tmux


# tmux attach -t exp
# docker exec -it hyoyoon-pretrain-nlp-exp1 bash
# conda activate test
# (conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia) : torch 섪치
# (python -m pip install -U pip setuptools wheel)
# (pip install -r requirements.txt) ## 한번 설치하면 컨테이어 내 conda (test)에서 계속 유지
# wandb login
# python src/main.py --config configs/exp.yaml


# 로컬 터미널에서 tmux 
# ssh jeaheekim@147.47.39.138
# ssh jeaheekim@147.47.134.100 -p 2222
# 이동 : for_pretrain/hyoyoon/exp_1
# docker attach hyoyoon-pretrain-nlp-exp1 bash
# tmux attach -t exp 
# python src/main.py --config configs/exp.yaml


# watch? nvidia-smi

## accelerate : 
# accelerate config default
# accelerate launch src/main.py --config configs/exp.yaml

# python -u -m src.main --config configs/bert_ebs256.yaml
# accelerate launch --num_processes 2 -m src.main --config configs/exp.yaml

import wandb 
from tqdm import tqdm
import os

import torch
import torch.nn
import omegaconf
from omegaconf import OmegaConf

from .utils import load_config #,set_logger
from .model import EncoderForClassification
from .data import get_dataloader

# torch.cuda.set_per_process_memory_fraction(11/24) -> 김재희 로컬과 신입생 로컬의 vram 맞추기 용도. 과제 수행 시 삭제하셔도 됩니다. 
# model과 data에서 정의된 custom class 및 function을 import합니다.

from transformers import set_seed


def train_iter(model, inputs, device): 
    ''' accumulation 위해서 1 step 기준이 아닌, micro batch 기준 forward만 수행'''
    inputs = {key : (value.to(device) if torch.is_tensor(value) else value) for key, value in inputs.items()} # 배치 텐서를 디바이스로 gpu
    outputs = model(**inputs)
    loss = outputs['loss']

    logits = outputs['logits'].detach() ## 통계 한꺼번에 내기 위해서 logits, labels 같이 return 
    labels = inputs['label']
    # backward 과정 삭제 
    #optimizer.zero_grad(set_to_none = True) ##
    #loss.backward()
    #optimizer.step()
    #wandb.log({'train_loss' : loss.item(), 'train_epoch': epoch})
    return loss, logits, labels

def valid_iter(model, inputs, device):
    '''
    1 step 평가 (model.eval() & no_grad() 상태에서 호출)
    '''
    inputs = {key : (value.to(device) if torch.is_tensor(value) else value) for key, value in inputs.items()} #
    outputs = model(**inputs)
    loss = outputs['loss']
    accuracy = calculate_accuracy(outputs['logits'], inputs['label'])    
    return loss, accuracy

def calculate_accuracy(logits, label):
    preds = logits.argmax(dim=-1)
    correct = (preds == label).sum().item()
    return correct / label.size(0)

## --- accelerator 이용 version ----
from accelerate import Accelerator

def main(configs : omegaconf.DictConfig) :
    # Set device - accelerate 에서
    set_seed(int(getattr(configs, 'seed', 42)))
    accum_steps = int(getattr(configs.train_config, 'accum_steps', 4))
    epochs = int(getattr(configs.train_config, 'epochs', 3))
    grad_clip = float(getattr(configs.train_config, "grad_clip", 0.0))
    log_every = int(getattr(configs.train_config, "log_every", 50))

    accelerator = Accelerator(gradient_accumulation_steps=accum_steps) # fp16/bf16, multi-GPU/DDP, cpu/xpu 자동 처리
    
    # wandb
    if accelerator.is_main_process:
        wandb.init(
            project=configs.logging.project,
            name=configs.logging.run_name,
            config=OmegaConf.to_container(configs, resolve=True)
        )

    # Load data
    train_loader = get_dataloader(configs.data, 'train')
    val_loader = get_dataloader(configs.data, 'valid')
    test_loader = get_dataloader(configs.data, 'test')

    # Load model
    model = EncoderForClassification(configs.model) # to_device 제거

    ## Set optimizer : 현재 adam으로 fix
    lr = float(getattr(configs.train_config, 'lr', 5e-5))
    weight_decay = float(getattr(configs.train_config, 'weight_decay', 0.0))
    optimizer = torch.optim.Adam(model.parameters(), # (p for p in model.parameters() if p.requires_grad)
                                 lr=lr, 
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _:1.0) ##

    out_dir = getattr(configs.train_config, "output_dir", "outputs/exp1")
    os.makedirs(out_dir, exist_ok=True)
    best_ckpt = os.path.join(out_dir, "best.pt")
    best_val_acc = -1.0

    ## accelerate prepare
    model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, scheduler
    )

    global_update = 0
    for epoch in range(1, epochs + 1):
        # ---- train + gradient accumulation ----
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0 # 창 단위 통계(업데이트 1회에 대응)

        # main 만 표시?
        pbar = tqdm(train_loader, desc=f"[train] epoch {epoch}", disable=not accelerator.is_main_process)

        for step, batch in enumerate(pbar, start=1):
            #loss, logits, labels = train_iter(model, batch, device) # batch는 이미 gpu에 올라감 , optimizer, device
            #(loss/ accum_steps).backward() # backward 누적

            # accelerate로 output
            #batch = move_to_device(batch, accelerator.device)
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs['loss'] / accum_steps ##
                accelerator.backward(loss)

                if grad_clip and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    scheduler.step()

            # 통계
            logits = outputs['logits'].detach()
            labels = batch['label']
            #preds = logits.detach().argmax(dim=-1)
            bs = labels.size(0)
            running_loss += loss.item() * bs
            running_correct += (logits.argmax(-1) == labels).sum().item() #(preds == labels).sum().item()
            running_total += bs
            #win_count += 1

            # accumulator로 로깅 간격 관리
            if accelerator.is_main_process and  (step % (accum_steps*log_every)) == 0:
                wandb.log({
                    "train/loss": running_loss / max(running_total, 1),
                    "train/acc": running_correct / max(running_total, 1),
                    "lr": optimizer.param_groups[0]["lr"],
                    "global_update": global_update,
                })
                pbar.set_postfix(loss=f"{running_loss/max(running_total,1):.4f}",
                                 acc=f"{running_correct/max(running_total,1):.4f}")
                running_loss = running_correct = running_total = 0 # 통계값 초기화
                #win_count = 0

        # Remainer 는 accumulator 내부에서 
        ### ---------------------------------------------------------------

        # ---- validation ----
        model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[valid] epoch {epoch}", leave=False, disable=not accelerator.is_main_process):
                #batch = move_to_device(batch, accelerator.device)
                outputs = model(**batch)
                loss = outputs['loss']
                logits = outputs['logits']
                labels = batch['label']

                gathered_logits = accelerator.gather_for_metrics(logits)
                gathered_labels = accelerator.gather_for_metrics(labels)

                preds = gathered_logits.argmax(dim=-1)
                correct += (preds == gathered_labels).sum().item() #int(acc * bs)
                total += gathered_labels.size(0)

                bs_local = labels.size(0)
                loss_sum_local = torch.tensor([loss.item()*bs_local],
                                              device=logits.device, dtype=torch.float32)
                loss_sum_global = accelerator.gather_for_metrics(loss_sum_local).sum().item()
                loss_sum += loss_sum_global
        val_loss = loss_sum / max(total, 1)
        val_acc = correct / max(total, 1)

        if accelerator.is_main_process:
            wandb.log({"val/loss": val_loss, "val/acc": val_acc, "epoch": epoch})
            print(f"[epoch {epoch}] val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # ---- save best ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state_dict = accelerator.get_state_dict(model)
            accelerator.save({'model':state_dict, 'val_acc':val_acc, 'epoch':epoch}, best_ckpt)
            #torch.save({"model": model.state_dict(), "val_acc": val_acc, "epoch": epoch}, best_ckpt)
            print(f"   saved best: {best_ckpt} (val_acc={val_acc:.4f})")
    
    accelerator.wait_for_everyone() ##
    
    ## Test with best checkpoint ----
    ckpt = torch.load(best_ckpt, map_location='cpu') ## 여기서만 체크포인트 로딩 시점의 디바이스 의존성 줄이기 위함 !
    accelerator.unwrap_model(model).load_state_dict(ckpt['model'])
    model.eval()

    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[test]", disable=not accelerator.is_main_process):
            outputs = model(**batch)
            loss = outputs['loss']
            logits = outputs['logits']
            labels = batch['label']

            gathered_logits = accelerator.gather_for_metrics(logits)
            gathered_labels = accelerator.gather_for_metrics(labels)

            preds = gathered_logits.argmax(dim=-1)
            correct += (preds == gathered_labels).sum().item() #int(acc * bs)
            total += gathered_labels.size(0)

            bs_local = labels.size(0)
            loss_sum_local = torch.tensor([loss.item()*bs_local],
                                              device=logits.device, dtype=torch.float32)
            loss_sum_global = accelerator.gather_for_metrics(loss_sum_local).sum().item()
            loss_sum += loss_sum_global

    test_loss = loss_sum / max(total, 1)
    test_acc = correct / max(total, 1)

    if accelerator.is_main_process:
        wandb.log({"test/loss": test_loss, "test/acc": test_acc})
        print(f"[BEST] val_acc={best_val_acc:.4f} | [TEST] loss={test_loss:.4f} acc={test_acc:.4f}")

    # validation for last epoch
    
    
if __name__ == "__main__" :
    configs = load_config()
    main(configs)