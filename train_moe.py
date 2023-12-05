import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
from tqdm import tqdm
from moe import MoE
from utils import AverageMeter, accuracy
from loss import LossComputer
import sparselearning
from sparselearning.core import Masking, CosineDecay, LinearDecay
from pytorch_transformers import AdamW, WarmupLinearSchedule

def run_epoch(epoch, model, optimizer, loader, loss_computer, logger, csv_logger, args,
              is_training, show_progress=False, log_every=50, scheduler=None, mask=None):
    """
    scheduler is only used inside this function if model is bert.
    """

    if is_training:
        model.train()
    else:
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader

    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(prog_bar_loader):

            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
   
            outputs, aux_loss  = model(x)

            loss_main = loss_computer.loss(outputs, y, g, is_training)

            if is_training:
 
                optimizer.zero_grad()
                total_loss = loss_main + aux_loss

                total_loss.backward()

                # if mask is not None:
                #     mask.step()
                # else:
                optimizer.step()
                mask.apply_mask()
                

            if is_training and (batch_idx+1) % log_every==0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()
                mask.print_density()
                
        if (not is_training) or loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()




def train_moe(model1,model2, criterion, dataset,
          logger, train_csv_logger, val_csv_logger, test_csv_logger,
          args, epoch_offset):
    
    #### model 1
    model1 = model1.cuda()



    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    train_loss_computer = LossComputer(
        criterion,
        is_robust=args.robust,
        dataset=dataset['train_data'],
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight)



    mask = None
    if args.sparse:
        decay = CosineDecay(args.death_rate, len(dataset['train_loader']) * int (args.n_epochs))
        mask = Masking(None, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay,
                    growth_mode=args.growth,
                    redistribution_mode=args.redistribution, args=args)
        mask.add_module(model1, sparse_init=args.sparse_init, density=args.density, train_loader=None)



    #### model 2
    model2 = model2.cuda()




    ### moe model


    model=MoE(model1=model1,model2=model2,input_size=224*224*3,num_experts=2, noisy_gating=True, k=2)
    model=model.cuda()
    
    ### optmizer 
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)






    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            factor=0.1,
            patience=5,
            threshold=0.0001,
            min_lr=0,
            eps=1e-08)
    else:
        scheduler = None

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)



    train_loss_computer = LossComputer(
        criterion,
        is_robust=args.robust,
        dataset=dataset['train_data'],
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight)




    best_val_acc = 0
    for epoch in range(epoch_offset, epoch_offset+ int (args.n_epochs) ):
        logger.write('\nEpoch [%d]:\n' % epoch)
        logger.write(f'Training:\n')
        run_epoch(
            epoch, model, optimizer,
            dataset['train_loader'],
            train_loss_computer,
            logger, train_csv_logger, args,
            is_training=True,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler,
            mask=mask)

        logger.write(f'\nValidation:\n')
        val_loss_computer = LossComputer(
            criterion,
            is_robust=args.robust,
            dataset=dataset['val_data'],
            step_size=args.robust_step_size,
            alpha=args.alpha)
        run_epoch(
            epoch, model, optimizer,
            dataset['val_loader'],
            val_loss_computer,
            logger, val_csv_logger, args,
            is_training=False,
            mask=mask)

        # Test set; don't print to avoid peeking
        if dataset['test_data'] is not None:
            print ("here is test acc!!")
            logger.write(f'\nTEST!:\n')

            test_loss_computer = LossComputer(
                criterion,
                is_robust=args.robust,
                dataset=dataset['test_data'],
                step_size=args.robust_step_size,
                alpha=args.alpha)
            run_epoch(
                epoch, model, optimizer,
                dataset['test_loader'],
                test_loss_computer,
                logger, test_csv_logger, args,
                is_training=False,
                mask=mask)

        # Inspect learning rates
        if (epoch+1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
                logger.write('Current lr: %f\n' % curr_lr)

        if args.scheduler and args.model != 'bert':
            if args.robust:
                val_loss, _ = val_loss_computer.compute_robust_loss_greedy(val_loss_computer.avg_group_loss, val_loss_computer.avg_group_loss)
            else:
                val_loss = val_loss_computer.avg_actual_loss
            scheduler.step(val_loss) #scheduler step to update lr at the end of epoch

        # if epoch % args.save_step == 0:
        #     torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

        if args.save_last:
            torch.save(model, os.path.join(args.log_dir, 'last_model.pth'))

        # if args.save_best:
        #     if args.robust or args.reweight_groups:
        #         curr_val_acc = min(val_loss_computer.avg_group_acc)
        #     else:
        #         curr_val_acc = val_loss_computer.avg_acc
        #     logger.write(f'Current validation accuracy: {curr_val_acc}\n')
        #     if curr_val_acc > best_val_acc:
        #         best_val_acc = curr_val_acc
        #         torch.save(model, os.path.join(args.log_dir, 'best_model.pth'))
        #         logger.write(f'Best model saved at epoch {epoch}\n')

        # infe using best val model
        if args.robust or args.reweight_groups:
            curr_val_acc = min(val_loss_computer.avg_group_acc)
        else:
            curr_val_acc = val_loss_computer.avg_acc

        if curr_val_acc > best_val_acc:
            best_val_acc = curr_val_acc
            best_model_dic=model.state_dict()
            print ("best valication acc",curr_val_acc )


        if args.automatic_adjustment:
            gen_gap = val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
            adjustments = gen_gap * torch.sqrt(train_loss_computer.group_counts)
            train_loss_computer.adj = adjustments
            logger.write('Adjustments updated\n')
            for group_idx in range(train_loss_computer.n_groups):
                logger.write(
                    f'  {train_loss_computer.get_group_name(group_idx)}:\t'
                    f'adj = {train_loss_computer.adj[group_idx]:.3f}\n')
        logger.write('\n')





    ### save sparse


    print ("load best model")
    model.load_state_dict(best_model_dic)


    test_loss_computer = LossComputer(
        criterion,
        is_robust=args.robust,
        dataset=dataset['test_data'],
        step_size=args.robust_step_size,
        alpha=args.alpha)

    run_epoch(
        epoch, model, optimizer,
        dataset['test_loader'],
        test_loss_computer,
        logger, test_csv_logger, args,
        is_training=False,
        mask=mask)

    curr_test_acc = min(test_loss_computer.avg_group_acc)
    print ("best test acc",curr_test_acc )
