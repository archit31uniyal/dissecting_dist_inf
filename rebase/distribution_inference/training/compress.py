from distribution_inference.training.compression_attack_cv_backup import *

def trainer_tune(model, criterion, optimizer, epoch, callback):
    # Dataloaders
    # train_loader, val_loader = get_cifar10(256, 256)
    # train_loader, val_loader = get_data('CIFAR10', clipped_class, clip_percentage, 256, 256)

    # loss function and optimiyer
    # loss_function = nn.CrossEntropyLoss() # your loss function, cross entropy works well for multi-class problems

    # optimizer, I've used Adadelta, as it wokrs well without any magic numbers
    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-4) # Using Karpathy's learning rate constant

    start_ts = time.time()

    losses = []
    batches = len(train_loader)
    val_batches = len(val_loader)

    # loop for every epoch (training + evaluation)
    # for epoch in range(epochs):
    total_loss = 0

    # progress bar (works in Jupyter notebook too!)
    progress = tqdm(enumerate(train_loader))

    # ----------------- TRAINING  -------------------- 
    # set model to training
    model.train()

    correct_pred = 0
    num_examples = 0    
    for i, data in progress:
        # X, y = data[0].to(device), data[1].to(device)
        X, y = data[0].to(device), data[1].to(device)
        # training step for single batch
        model.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.mean().backward()
        if callback:
            callback()
        optimizer.step()

        # getting training quality data
        current_loss = loss.item()
        total_loss += current_loss

        predicted_classes = torch.max(outputs, 1)[1]
        correct_pred += (predicted_classes == y).sum()
        num_examples += y.size()[0]

        # updating progress bar
        # val_loss, val_acc = test(model, val_loader)
        # model.train()

        progress.set_description("Epoch: {}, Loss: {:.4f}, Train accuracy: {:.4f}".format(epoch, total_loss/(i+1), correct_pred/(num_examples)))
        # print("Epoch: {}, Loss: {:.4f}, Train accuracy: {:.4f}".format(epoch, total_loss/(i+1), correct_pred/(num_examples)))
        # logger.write("Epoch: {}, Loss: {:.4f}, Train accuracy: {:.4f}, Val_loss: {:.4f}, Val_accuracy: {:.4f}".format(epoch, total_loss/(i+1), correct_pred/(num_examples), val_loss, val_acc))
        data[0] = data[0].detach().cpu()
        data[1] = data[1].detach().cpu()

    # releasing unecessary memory in GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # print(f"{'-'*10} Cuda memory cleaned {'-'*10}")

def evaluate_tune(model):
    val_losses = 0
    precision, recall, f1, accuracy = [], [], [], []
    
    # _, val_loader = get_cifar10(256, 256)

    # _, val_loader = get_data('CIFAR10', clipped_class, clip_percentage, 256, 256)
    
    # prob_scores = []

    val_batches = len(val_loader)
    criterion = nn.CrossEntropyLoss()
    # set model to evaluating (testing)
    model.eval()
    with torch.no_grad():
        tespreds, tesactuals  = [], [],
        num_examples, correct_testpreds = 0, 0
        for i, data in enumerate(val_loader):
            # X, y = data[0].to(device), data[1].to(device)
            X, y = data[0].to(device), data[1].to(device)
            
            outputs = model(X) # this get's the prediction from the network

            # prob_scores.append(torch.nn.Softmax()(torch.mean(outputs, 0)))

            val_losses += criterion(outputs, y)

            predicted_classes = torch.max(outputs, 1)[1] # get class from network's prediction
            correct_testpreds += (predicted_classes == y).sum()
            num_examples += y.size()[0]
            # print(num_examples)

            tespreds += list(predicted_classes.cpu().numpy())
            tesactuals += list(y.cpu().numpy())

            # print(torch.min(X), torch.max(X))
            # print(y.shape)
            # for l in list(set(predicted_classes)):
            #     tespreds.append(predicted_classes[l].item())
            #     tesactuals.append(y[l].cpu())
            # calculate P/R/F1/A metrics for batch
            for acc, metric in zip((precision, recall, f1), 
                                    (precision_score, recall_score, f1_score)):
                acc.append(
                    calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                )
            data[0] = data[0].detach().cpu()
            data[1] = data[1].detach().cpu()
            # print(predicted_classes.shape)
            # print(f"Validation loss: {val_losses/val_batches}    Test Accuracy: {correct_testpreds.float()/num_examples}")
            # exit(0)

        conf_mat = sklearn.metrics.confusion_matrix(tesactuals, tespreds, labels = classes)
        # conf_scores = torch.mean(torch.stack(prob_scores), 0)
        
        # releasing unecessary memory in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"{'-'*10} Cuda memory cleaned {'-'*10}")

    print(f"Validation loss: {val_losses/val_batches}    Test Accuracy: {correct_testpreds.float()/num_examples}")
    # logger.write(f"Test Accuracy: {correct_testpreds.float()/num_examples}")
    print(conf_mat)
    # logger.write(conf_scores)
    # print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
    # print(f"Test Accuracy: {correct_testpreds.float()/num_examples}")
    # print_scores(precision, recall, f1, val_batches)

    return val_losses/val_batches, correct_testpreds.float()/num_examples


def compress_models(model, loaders, trainer = trainer_tune, evaluate = evaluate_tune):
    global train_loader, val_loader
    
    train_loader, val_loader = loaders[0], loaders[1]
    
    for batch in train_loader:
        # print(batch.keys())
        dummy_input = [batch[0].to(device), batch[1].to(device)]
        # print(batch)
        break
    
    model = model.to(device)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss() # your loss function, cross entropy works well for multi-class problems

    # optimizer, I've used Adadelta, as it wokrs well without any magic numbers
    optimizer = torch.optim.AdamW(model.parameters(), lr=16e-4) # Using Karpathy's learning rate constant
    # 8e-4 -> 77.3
    # 6e-4 -> 76.3
    # params you need to specify:
    epochs = 5
    batch_size = 32
    
    # logger.write(f"\n{'--'*10} Compressing without class 6 {'--'*10}\n")

    config_list = [{
            'sparsity': 0.5,
            'op_types': ['Conv2d']  
        }]

    pruner = AutoCompressPruner(
                model, config_list, trainer=trainer, evaluator=evaluate,
                dummy_input=dummy_input, num_iterations=2, optimize_mode='maximize', base_algo='l1',
                cool_down_rate=0.9, admm_num_iterations=2, admm_epochs_per_iteration = 3, experiment_data_dir=f'/p/compressionleakage/logs/Compressed/compression_cv/models/celeba')
    model = pruner.compress()

    # torch.save(model.state_dict(), f'{directory}/models/celeba/')
    val_loss, val_acc = evaluate_tune(model)
    # pruner.export_model(f'{directory}/models/celeba/resnet50/adv/Male/0.5/ft_train_0.5000/32.ch')
    return model, (val_loss, val_acc)


    
