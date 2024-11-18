from utils.metrics import cmu_weighted_accuracy, cmu_micro_f1, cmu_accuracy, cmu_macro_f1, cmu_uar
from utils.metrics import emo_accuracy, emo_macro_f1, emo_micro_f1, emo_uar
from sklearn.utils.class_weight import compute_class_weight
import random
import numpy as np
import torch
from tqdm import tqdm

# singe-label

def train_one_epoch_sl(dataloader, train_writer, epoch, optimizer, model, device, criterion, modality, scheduler_type, scheduler):
    predictions, targets, processed_size, running_loss, loss,  = list(), list(), 0., 0., 0.
    iters = len(dataloader)
    for i, data in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        if modality == 'a':
            _, audio, labels = data
            pred = model(audio.to(device))
            labels = labels.type(torch.LongTensor).to(device)
        elif modality == 'v':
            _, video, labels = data
            pred = model(video.to(device))
            labels = labels.type(torch.LongTensor).to(device)
        elif modality == 'avt':
            audio, video, text, labels = data
            pred = model(audio.to(device), video.to(device), text.to(device))
            labels = labels.type(torch.LongTensor).to(device)
        elif modality == 't':
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            pred = outputs.logits
        loss = criterion(pred,labels)
        loss.backward()
        optimizer.step()
        if scheduler_type == 'CosineAnnealingWarmRestarts':
            scheduler.step(epoch + i / iters)

        pred = torch.argmax(pred, dim=1).cpu().numpy()
        true = labels.cpu().numpy()
        predictions.extend(pred)
        targets.extend(true)

        processing_size = len(labels)
        processed_size += processing_size
        
        running_loss += loss.item() * processing_size

    avg_loss = running_loss / processed_size
    uar = emo_uar(targets, predictions)
    acc = emo_accuracy(targets, predictions)
    mf1 = emo_macro_f1(targets, predictions)
    wf1 = emo_micro_f1(targets, predictions)

    train_writer.add_scalar('loss', avg_loss, epoch)
    train_writer.add_scalar('uar', uar, epoch)
    train_writer.add_scalar('mf1', mf1, epoch)
    train_writer.add_scalar('wf1', wf1, epoch)
    train_writer.add_scalar('acc', acc, epoch)
    
    return avg_loss, uar, acc, mf1, wf1

def val_one_epoch_sl(dataloader, val_writer, epoch, model, device, criterion, modality):
    predictions, targets, processed_size, running_loss, loss,  = list(), list(), 0., 0., 0.
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            if modality == 'a':
                _, audio, labels = data
                pred = model(audio.to(device))
                labels = labels.type(torch.LongTensor).to(device)
            elif modality == 'v':
                _, video, labels = data
                pred = model(video.to(device))
                labels = labels.type(torch.LongTensor).to(device)
            elif modality == 'avt':
                audio, video, text, labels = data
                pred = model(audio.to(device), video.to(device), text.to(device))
                labels = labels.type(torch.LongTensor).to(device)
            elif modality == 't':
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                labels = data['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                pred = outputs.logits

            loss = criterion(pred,labels)

            processing_size = len(labels)
            processed_size += processing_size
            
            running_loss += loss.item() * processing_size
            pred = torch.argmax(pred, dim=1)
            pred = pred.cpu().numpy()
            true = labels.cpu().numpy()
            predictions.extend(pred)
            targets.extend(true)
            
    avg_loss = running_loss / processed_size
    
    uar = emo_uar(targets, predictions)
    acc = emo_accuracy(targets, predictions)
    mf1 = emo_macro_f1(targets, predictions)
    wf1 = emo_micro_f1(targets, predictions)

    val_writer.add_scalar('loss', avg_loss, epoch)
    val_writer.add_scalar('uar', uar, epoch)
    val_writer.add_scalar('acc', acc, epoch)
    val_writer.add_scalar('mf1', mf1, epoch)
    val_writer.add_scalar('wf1', wf1, epoch)

    return avg_loss, uar, acc, mf1, wf1

# binary-label

def train_one_epoch_bl(dataloader, train_writer, epoch, optimizer, model, device, criterion, modality, scheduler_type, scheduler):
    predictions, targets, processed_size, running_loss, loss,  = list(), list(), 0., 0., 0.
    iters = len(dataloader)
    for i, data in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        if modality == 'a':
            _, audio, labels = data
            pred = model(audio.to(device))
            labels = labels.type(torch.LongTensor).to(device)
        elif modality == 'v':
            _, video, labels = data
            pred = model(video.to(device))
            labels = labels.type(torch.LongTensor).to(device)
        elif modality == 'avt':
            audio, video, text, labels = data
            pred = model(audio.to(device), video.to(device), text.to(device))
            labels = labels.float().to(device)
        elif modality == 't':
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            pred = outputs.logits
        pred = torch.sigmoid(pred)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        if scheduler_type == 'CosineAnnealingWarmRestarts':
            scheduler.step(epoch + i / iters)

        pred = pred.detach().cpu().numpy()
        pred = np.where(pred >= 0.5, 1, 0)
        true = labels.cpu().numpy()
        predictions.extend(pred)
        targets.extend(true)

        processing_size = len(audio)
        processed_size += processing_size
        
        running_loss += loss.item() * processing_size

    avg_loss = running_loss / processed_size
    wacc = cmu_weighted_accuracy(targets, predictions)
    wf1 = cmu_micro_f1(targets, predictions)
    acc = cmu_accuracy(targets, predictions)
    mf1 = cmu_macro_f1(targets, predictions)
    uar = cmu_uar(targets, predictions)

    train_writer.add_scalar('loss', avg_loss, epoch)
    train_writer.add_scalar('wacc', wacc, epoch)
    train_writer.add_scalar('wf1', wf1, epoch)
    train_writer.add_scalar('acc', acc, epoch)
    train_writer.add_scalar('mf1', mf1, epoch)
    train_writer.add_scalar('uar', uar, epoch)
    
    return avg_loss, wacc, wf1, acc, mf1, uar

def val_one_epoch_bl(dataloader, val_writer, epoch, model, device, criterion, modality):
    predictions, targets, processed_size, running_loss, loss,  = list(), list(), list(), 0., 0., 0.
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            if modality == 'a':
                audio, labels = data
                pred = model(audio.to(device))
                labels = labels.type(torch.LongTensor).to(device)
            elif modality == 'v':
                video, labels = data
                pred = model(video.to(device))
                labels = labels.type(torch.LongTensor).to(device)
            elif modality == 'avt':
                audio, video, text, labels = data
                pred = model(audio.to(device), video.to(device), text.to(device))
                labels = labels.float().to(device)
            elif modality == 't':
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                labels = data['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                pred = outputs.logits
            pred = torch.sigmoid(pred)
            loss = criterion(pred, labels)

            processing_size = len(audio)
            processed_size += processing_size
            
            running_loss += loss.item() * processing_size
            pred = pred.detach().cpu().numpy()
            pred = np.where(pred >= 0.5, 1, 0)
            true = labels.cpu().numpy()
            predictions.extend(pred)
            targets.extend(true)
            
    avg_loss = running_loss / processed_size
    
    wacc = cmu_weighted_accuracy(targets, predictions)
    wf1 = cmu_micro_f1(targets, predictions)
    acc = cmu_accuracy(targets, predictions)
    mf1 = cmu_macro_f1(targets, predictions)
    uar = cmu_uar(targets, predictions)

    val_writer.add_scalar('loss', avg_loss, epoch)
    val_writer.add_scalar('wacc', wacc, epoch)
    val_writer.add_scalar('wf1', wf1, epoch)
    val_writer.add_scalar('acc', acc, epoch)
    val_writer.add_scalar('mf1', mf1, epoch)
    val_writer.add_scalar('uar', uar, epoch)

    return avg_loss, wacc, wf1, acc, mf1, uar

def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_weights(train_dataloader):
    class_counts = {}
    trian_labels = []
    for data in tqdm(train_dataloader):
        if len(data) == 3:
            _, _, labels = data
        elif len(data) == 4:
            _, _, _, labels = data
        for label in labels.numpy():
            trian_labels.append(label)
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1
    class_weights = compute_class_weight('balanced', classes=np.unique(trian_labels), y=trian_labels)
    print('Class weights: ', class_weights)
    print("Class_counts:", class_counts)
    return class_weights

def get_binary_weights(train_dataloader):
    trian_labels = []
    
    for data in tqdm(train_dataloader):
        if len(data) == 3:
            _, _, labels = data
        elif len(data) == 4:
            _, _, _, labels = data
        trian_labels.append(labels)
    
    num_positives = torch.sum(torch.Tensor(trian_labels), dim=0)
    num_negatives = len(trian_labels) - num_positives
    pos_weight  = num_negatives / num_positives
    print('Binary class weights: ', pos_weight)
    return pos_weight