import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import re
import os
import math
import copy

import numpy as np
from tqdm import tqdm

from utils import AverageMeter, accuracy
from loss import LossComputer
import sparselearning
from sparselearning.core import Masking, CosineDecay, LinearDecay
from pytorch_transformers import AdamW, WarmupLinearSchedule
from models import model_attributes




from data.data import dataset_attributes, shift_types, prepare_data, log_data
from utils import set_seed, Logger, CSVBatchLogger, log_args

from variable_width_resnet import resnet50vw, resnet18vw, resnet10vw

def run_epoch(epoch, model, optimizer, loader, loss_computer, logger, csv_logger, args,
              is_training, show_progress=False, log_every=50, scheduler=None, mask=None):
    """
    scheduler is only used inside this function if model is bert.
    """

    if is_training:
        model.train()
        if args.model == 'bert':
            model.zero_grad()
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
            if args.model == 'bert':
                input_ids = x[:, :, 0]
                input_masks = x[:, :, 1]
                segment_ids = x[:, :, 2]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    labels=y
                )[1] # [1] returns logits
            else:
                outputs = model(x)

            loss_main = loss_computer.loss(outputs, y, g, is_training)

            if is_training:
                if args.model == 'bert':
                    loss_main.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scheduler.step()
                    optimizer.step()
                    model.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss_main.backward()

                    if mask is not None:
                        mask.step()
                    else:
                        optimizer.step()

            if is_training and (batch_idx+1) % log_every==0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()

        if (not is_training) or loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()

def model_files_filter(model_files,filter_itrs=["best"]):
    new_files=[]
    for filter_itr in filter_itrs:
        for  model_file in model_files:
            if filter_itr in model_file:
                new_files.append(model_file)
    return new_files



def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)



def setup_model(args,dataset):

    train_data=dataset['train_data']

    pretrained = not args.train_from_scratch
    n_classes =train_data.n_classes

    if model_attributes[args.model]['feature_type'] in ('precomputed', 'raw_flattened'):
        assert pretrained
        # Load precomputed features
        d = train_data.input_size()[0]
        model = nn.Linear(d, n_classes)
        model.has_aux_logits = False
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'resnet34':
        model = torchvision.models.resnet34(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'wideresnet50':
        model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model =='resnet50vw':
        assert not pretrained
        assert args.resnet_width is not None
        model = resnet50vw(args.resnet_width, num_classes=n_classes)
    elif args.model =='resnet18vw':
        assert not pretrained
        assert args.resnet_width is not None
        model = resnet18vw(args.resnet_width, num_classes=n_classes)

    return model


def get_model_params(model):
    params = {}
    for name in model.state_dict():
        params[name] = copy.deepcopy(model.state_dict()[name])
    return params

def set_model_params(model, model_parameters):
    model.load_state_dict(model_parameters)



def train_merge_lp(model, criterion, dataset,
          logger, train_csv_logger, val_csv_logger, test_csv_logger,
          args, epoch_offset):

    val_worst_acc=[]
    test_worst_acc=[]

    val_average_acc=[]
    test_average_acc=[]

    

    epoch=0
    ######################## infe
    model_files = os.listdir(args.log_dir)

    model_files=model_files_filter(model_files)
    model_files = sorted_nicely(model_files)
#     model_files=list(reversed(model_files))

    model_files= ['best_sparse_0.05.pth', 'best_sparse_0.1.pth', 'best_sparse_0.2.pth', 'best_sparse_0.3.pth', 'best_sparse_0.4.pth', 'best_sparse_0.5.pth', 'best_sparse_0.6.pth', 'best_sparse_0.7.pth','dense_model.pth']

    # model_files= ['best_sparse_0.05.pth', 'best_sparse_0.1.pth', 'best_sparse_0.2.pth']
    print ("model_files",model_files)


    free_tickets=[]

    model_files=model_files
    for model_file in model_files:
        print ("model_file",model_file)
        ## init model

        model = torch.load(os.path.join(args.log_dir, model_file))



        ####  valication
        logger.write(f'\nValidation:\n')

        val_loss_computer = LossComputer(
            criterion,
            is_robust=args.robust,
            dataset=dataset['val_data'],
            step_size=args.robust_step_size,
            alpha=args.alpha)

        run_epoch(
            epoch, model, None,
            dataset['val_loader'],
            val_loss_computer,
            logger, val_csv_logger, args,
            is_training=False,
            mask=None)

        ####  test acc
        logger.write(f'\nTEST!:\n')

        test_loss_computer = LossComputer(
            criterion,
            is_robust=args.robust,
            dataset=dataset['test_data'],
            step_size=args.robust_step_size,
            alpha=args.alpha)
        run_epoch(
            epoch, model, None,
            dataset['test_loader'],
            test_loss_computer,
            logger, test_csv_logger, args,
            is_training=False,
            mask=None)

        print ("===============")

        
        
        free_tickets.append(model)

        val_worst = min(val_loss_computer.avg_group_acc)
        test_worst = min(test_loss_computer.avg_group_acc)

        val_average=val_loss_computer.avg_acc.item()
        test_average=test_loss_computer.avg_acc.item()

        val_worst_acc.append(val_worst)
        test_worst_acc.append(test_worst )


        val_average_acc.append(val_average )
        test_average_acc.append(test_average)


        print ("test_worst_acc",test_worst_acc)
        print ("val_worst",val_worst_acc)

        print ("val_average",val_average_acc )
        print ("test_average",test_average_acc )



    print ("inference done")
    print ("\n")



    #### help function
    def get_or_averagte_moving(decay):

        def function(free_tickets,rangelist,decay=decay):
            print ("ensemble by OR weights averagte_moving ")
            print ("decay",decay)
            ensemble_flag="moving_"+str(decay)
            params = {}
            pre_params = {}
            for name in free_tickets[0].state_dict():
                pre_params[name] = copy.deepcopy(free_tickets[0].state_dict()[name])

            rangelist=list(rangelist)[1:]
            for name in free_tickets[0].state_dict():
                for i in rangelist:
                    params[name]=copy.deepcopy(free_tickets[i].state_dict()[name]* decay+pre_params[name]* (1 - decay))
                    pre_params[name]=params[name]                    
                                                
            return params,ensemble_flag
        return function



    def average_greedy_backfront(use_free_tickets,start,new_indx):
        
        new_indx=np.array(new_indx)
        val_worst=val_worst_acc[start]


        print ("\n")
        print ("creating ensemble model")


        print ("number of tickets",len(use_free_tickets))

        print ("begin greedy search")


        if args.eval_by_ave:
            best_acc=val_average_acc[start]
        else:
            best_acc=val_worst



        current_ticket=use_free_tickets[0]
        
        
        to_pend_ind=[0]
        best_decay_all=[]

    #     for i in range(1,len(use_free_tickets)): 


        for i in range(1,len(use_free_tickets)  ):  
            
            
            new_ticket=use_free_tickets[i]
            
            
            
            current_tickets_model=[ current_ticket,new_ticket]

            ### acc 1

            val_loss_computer = LossComputer(
                criterion,
                is_robust=args.robust,
                dataset=dataset['val_data'],
                step_size=args.robust_step_size,
                alpha=args.alpha)
            run_epoch(
                epoch, current_ticket, None,
                dataset['val_loader'],
                val_loss_computer,
                logger, val_csv_logger, args,
                is_training=False,
                mask=None)
            current_ticket_acc = min(val_loss_computer.avg_group_acc)

            print ("current_ticket_acc",current_ticket_acc)

            print ("***")

           ### acc 2
            val_loss_computer = LossComputer(
                criterion,
                is_robust=args.robust,
                dataset=dataset['val_data'],
                step_size=args.robust_step_size,
                alpha=args.alpha)
            run_epoch(
                epoch, new_ticket, None,
                dataset['val_loader'],
                val_loss_computer,
                logger, val_csv_logger, args,
                is_training=False,
                mask=None)
            new_ticket_acc = min(val_loss_computer.avg_group_acc)
            print ("new_ticket_acc",new_ticket_acc)
            
            current_ticket_ind=copy.deepcopy(to_pend_ind)
            current_ticket_ind.append(i)
                
            

            print ("================")

            print ("search inex",i,"current search tickets len",len(current_ticket_ind),new_indx[np.array(current_ticket_ind)])

            
            
            moving_value=args.decay_value_list
            # init model 
            best_moving_acc=0
            best_decay=0

            
            print ("beging search moving value")
            for i in range(len(moving_value)):
                method=get_or_averagte_moving(moving_value[i])
                
                ensemble_model = setup_model(args,dataset)
                ensemble_model=ensemble_model.cuda()
                params,ensemble_flag = method(current_tickets_model,range(len(current_tickets_model)))
                set_model_params(ensemble_model, params)



                ### prune
            
                # if density!=None:

                #     weight_abs = []

                #     for name, weight in ensemble_model.named_parameters():
                #         if name not in current_mask: continue
                #         weight_abs.append(torch.abs(weight))

                #     # Gather all scores in a single vector and normalise
                #     all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
                #     num_params_to_keep = int(len(all_scores) * density)

                #     threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
                #     acceptable_score = threshold[-1]


                #     for name, weight in ensemble_model.named_parameters():
                #         if name not in current_mask: continue
                #         current_mask[name][:] = ((torch.abs(weight)) >= acceptable_score).float()


                #     for name, tensor in ensemble_model.named_parameters():
                #         if name in current_mask:
                #             tensor.data = tensor.data * current_mask[name]



        #        "update_bn"

                torch.optim.swa_utils.update_bn(dataset["train_loader"], ensemble_model,"cuda")
        #         print ("update_bn done")



                val_loss_computer = LossComputer(
                    criterion,
                    is_robust=args.robust,
                    dataset=dataset['val_data'],
                    step_size=args.robust_step_size,
                    alpha=args.alpha)
                run_epoch(
                    epoch, ensemble_model, None,
                    dataset['val_loader'],
                    val_loss_computer,
                    logger, val_csv_logger, args,
                    is_training=False,
                    mask=None)


                if args.eval_by_ave:
                    test_moving_acc=val_loss_computer.avg_acc
                else:
                    test_moving_acc = min(val_loss_computer.avg_group_acc)



                
                print('* Test Accurayc = {}'.format(test_moving_acc))
                
                if test_moving_acc > best_moving_acc:
                    best_ensemble_model=copy.deepcopy(ensemble_model)
                    
                    best_moving_acc=test_moving_acc
                    best_decay=moving_value[i]
                
                    print ("best decay",best_decay,"at",best_moving_acc)
                
                
                print ("\n")
                

            if best_moving_acc > best_acc:
                best_acc=best_moving_acc
                print ("best_acc",best_moving_acc)

                current_ticket=copy.deepcopy(best_ensemble_model)
                
                to_pend_ind=current_ticket_ind
                best_decay_all.append(best_decay)
            
    
            

        # save_checkpoint({
        # 'state_dict': current_ticket.state_dict(),
        # }, is_SA_best=False, save_path=args.checkpoint,filename="model_Avereage_"+str(len(moving_value))+".pth.tar")
        
            

        test_loss_computer = LossComputer(
            criterion,
            is_robust=args.robust,
            dataset=dataset['test_data'],
            step_size=args.robust_step_size,
            alpha=args.alpha)

        run_epoch(
            epoch, current_ticket, None,
            dataset['test_loader'],
            test_loss_computer,
            logger, val_csv_logger, args,
            is_training=False,
            mask=None)
        best_acc = min(val_loss_computer.avg_group_acc)
  
        now_average_acc=val_loss_computer.avg_acc

    

        print('original worse acc',test_worst_acc[start])
        print ("now worse acc",best_acc)


        print('original average acc',test_average_acc[start])
        print ("now average acc",now_average_acc)


        
        print ("best decays",best_decay_all)
        print ("best models",new_indx[np.array(to_pend_ind)])
        print ("**************")



    ####
    for i in range(1,len(free_tickets)+1):
        new_indx=[]
        start=i-1
        skip_num=1
        new_indx.append(start)
        for _ in range(len(free_tickets)):
        #     print ("skip_num",skip_num,i,i-skip_num,i+skip_num)

            if start-skip_num>=0:new_indx.append(start-skip_num)
            if start+skip_num<len(free_tickets):new_indx.append(start+skip_num)

            skip_num+=1
            
        print ("new_indx",new_indx)

        use_free_tickets=[free_tickets[i] for i in new_indx]
        
        
        
    #     print ("reverse")
    #     use_free_tickets=list(reversed(use_free_tickets))

        average_greedy_backfront(use_free_tickets,start,new_indx)