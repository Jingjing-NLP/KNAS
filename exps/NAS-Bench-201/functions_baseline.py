#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08 #
#####################################################
import time, torch
from procedures   import prepare_seed, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from config_utils import dict2config
from log_utils    import AverageMeter, time_string, convert_secs2time
from models       import get_cell_based_tiny_net


__all__ = ['evaluate_for_seed', 'pure_evaluate']


def pure_evaluate(xloader, network, criterion=torch.nn.CrossEntropyLoss()):
  data_time, batch_time, batch = AverageMeter(), AverageMeter(), None
  losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
  latencies = []
  network.eval()
  with torch.no_grad():
    end = time.time()
    for i, (inputs, targets) in enumerate(xloader):
      targets = targets.cuda(non_blocking=True)
      inputs  = inputs.cuda(non_blocking=True)
      data_time.update(time.time() - end)
      # forward
      features, logits = network(inputs)
      loss             = criterion(logits, targets)
      batch_time.update(time.time() - end)
      if batch is None or batch == inputs.size(0):
        batch = inputs.size(0)
        latencies.append( batch_time.val - data_time.val )
      # record loss and accuracy
      prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
      losses.update(loss.item(),  inputs.size(0))
      top1.update  (prec1.item(), inputs.size(0))
      top5.update  (prec5.item(), inputs.size(0))
      end = time.time()
  if len(latencies) > 2: latencies = latencies[1:]
  return losses.avg, top1.avg, top5.avg, latencies



def procedure(xloader, network, criterion, scheduler, optimizer, mode, grad=False):
  losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
  if mode == 'train'  : network.train()
  elif mode == 'valid': network.eval()
  else: raise ValueError("The mode is not right : {:}".format(mode)) 
  grads = {}
  
  data_time, batch_time, end = AverageMeter(), AverageMeter(), time.time()
  for i, (inputs, targets) in enumerate(xloader):
    #if  > 50: break
    #if mode != 'train': break
    if mode == 'train': scheduler.update(None, 1.0 * i / len(xloader))

    targets = targets.cuda(non_blocking=True)
    if mode == 'train': optimizer.zero_grad()
    # forward
    features, logits = network(inputs)
    loss             = criterion(logits, targets)
    # backward
    #print(int(targets[0].data))
    #if int(targets[0].data) != 1: continue
    if mode == 'train':
      loss.backward()
      import copy
      #if not grad: return 0, 0, 0,batch_time.sum
      index_grad = 0
      '''for name, param in network.named_parameters():
           #print(param.grad.view(-1).data)
           if param.grad == None:
                    #print(name)
                    continue
           #if param.grad.view(-1)[0] == 0 and param.grad.view(-1)[1] == 0: continue #print(name)
           #print(i)
           index_grad = name
           #index_grad += 1
           if name in grads: grads[name].append(copy.copy(param.grad.view(-1).data[0:100]))
           else: grads[name]=[copy.copy(param.grad.view(-1).data[0:100])]
      #print(index_grad)
      if len(grads[index_grad]) == 50:
                     conv = 0
                     maxconv = 0
                     minconv = 0
                     lower_layer = 1
                     top_layer = 1
                     para = 0
                     for name in grads:
                        #print(name)
                        for i in range(50):#nt(self.grads[name][0].size()[0])):
                               #if len(grads[name])!=: print(name)
                            for j in range(50):
                               if i == j: continue
                               grad1 = torch.tensor(grads[name][j],dtype=torch.float)#torch.tensor([grads[name][j][k] for j in range(25)],dtype=torch.float)
                               grad2 = torch.tensor(grads[name][i],dtype=torch.float)#torch.tensor([grads[name][j][k] for j in range(25,50)],dtype=torch.float)
                               grad1 = grad1# - grad1.mean()
                               grad2 = grad2# - grad2.mean()
                               #print(grad1,grad2)      
                               #print(torch.dot(torch.tensor(grad1, dtype=torch.float), torch.tensor(grad1,dtype=torch.float)))      
                               #if maxconv == None and minconv == None: 
                               #        maxconv = torch.dot(grad1, grad2)#i#torch.tensor(grad1, dtype=torch.float), torch.tensor(grad1,dtype=torch.float))
                               #        minconv = torch.dot(grad1, grad2)#(torch.tensor(grad1, dtype=torch.float), torch.tensor(grad1,dtype=torch.float))
                               #maxconv = max(maxconv,torch.dot(grad1, grad2))#torch.tensor(grad1, dtype=torch.float), torch.tensor(grad1,dtype=torch.float)), maxconv)
                               #minconv = min(minconv, torch.dot(grad1, grad2))#torch.tensor(grad1, dtype=torch.float), torch.tensor(grad1,dtype=torch.float)), minconv)
                               #if '.1.' in name: lower_layer += torch.dot(torch.tensor(grad1, dtype=torch.float), torch.tensor(grad1,dtype=torch.float))
                               #if '.11.' in name: top_layer += torch.dot(torch.tensor(grad1, dtype=torch.float), torch.tensor(grad1,dtype=torch.float))
                               conv += torch.dot(grad1, grad2)/2500#torch.tensor(grad1, dtype=torch.float), torch.tensor(grad1,dtype=torch.float))#i#/i1.0*self.grads[name][0].size()[0]
                               para+=1
                     conv /= para
                     print(conv, maxconv, minconv)# top_layer/lower_layer)
                     print("endddddddddd")
                     break'''
      #else: print(name)
      # optimizer.step()
      optimizer.step()
    # record loss and accuracy
    prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
    losses.update(loss.item(),  inputs.size(0))
    top1.update  (prec1.item(), inputs.size(0))
    top5.update  (prec5.item(), inputs.size(0))
    # count time
    batch_time.update(time.time() - end)
    end = time.time()
  if mode == 'train': return 0,0,0, batch_time.sum#return conv, maxconv, minconv, batch_time.sum #conv, maxconv, minconv
  else:  return losses.avg, top1.avg, top5.avg,batch_time.sum



def evaluate_for_seed(arch_config, config, arch, train_loader, valid_loaders, seed, logger):

  prepare_seed(seed) # random seed
  net = get_cell_based_tiny_net(dict2config({'name': 'infer.tiny',
                                             'C': arch_config['channel'], 'N': arch_config['num_cells'],
                                             'genotype': arch, 'num_classes': config.class_num}
                                            , None)
                                 )
  #net = TinyNetwork(arch_config['channel'], arch_config['num_cells'], arch, config.class_num)
  flop, param  = get_model_infos(net, config.xshape)
  logger.log('Network : {:}'.format(net.get_message()), False)
  logger.log('{:} Seed-------------------------- {:} --------------------------'.format(time_string(), seed))
  logger.log('FLOP = {:} MB, Param = {:} MB'.format(flop, param))
  # train and valid
  optimizer, scheduler, criterion = get_optim_scheduler(net.parameters(), config)
  network, criterion = torch.nn.DataParallel(net).cuda(), criterion.cuda()
  # start training
  start_time, epoch_time, total_epoch = time.time(), AverageMeter(), config.epochs + config.warmup
  train_losses, train_acc1es, train_acc5es, valid_losses, valid_acc1es, valid_acc5es = {}, {}, {}, {}, {}, {}
  train_conv, train_maxconv, train_minconv = {}, {}, {}
  train_times , valid_times = {}, {}
  for epoch in range(30):#total_epoch):
    scheduler.update(epoch, 0.0)

    train_loss, train_acc1, train_acc5, train_tm  = procedure(train_loader, network, criterion, scheduler, optimizer, 'train', grad=(epoch==0))
    train_losses[epoch] = train_loss
    train_acc1es[epoch] = train_acc1 
    train_acc5es[epoch] = train_acc5
    train_times [epoch] = train_tm
    #train_conv[epoch] = conv
    #train_maxconv[epoch] = maxconv
    #train_minconv[epoch] = minconv
    with torch.no_grad():
      for key, xloder in valid_loaders.items():
        valid_loss, valid_acc1, valid_acc5, valid_tm = procedure(xloder  , network, criterion,      None,      None, 'valid')
        valid_losses['{:}@{:}'.format(key,epoch)] = valid_loss
        valid_acc1es['{:}@{:}'.format(key,epoch)] = valid_acc1 
        valid_acc5es['{:}@{:}'.format(key,epoch)] = valid_acc5
        valid_times ['{:}@{:}'.format(key,epoch)] = valid_tm

    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()
    need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.avg * (total_epoch-epoch-1), True) )
    logger.log('{:} {:} epoch={:03d}/{:03d} :: Train [loss={:.5f}, acc@1={:.2f}%, acc@5={:.2f}%] Valid [loss={:.5f}, acc@1={:.2f}%, acc@5={:.2f}%]'.format(time_string(), need_time, epoch, total_epoch, train_loss, train_acc1, train_acc5, valid_loss, valid_acc1, valid_acc5))
    #logger.log(conv, maxconv, minconv)
    info_seed = {'flop' : flop,
               #'param': param,
               'channel'     : arch_config['channel'],
               'num_cells'   : arch_config['num_cells'],
               'config'      : config._asdict(),
               'total_epoch' : total_epoch ,
               'train_losses': train_losses,
               'train_acc1es': train_acc1es,
               'train_acc5es': train_acc5es,
               'train_times' : train_times,
               'valid_losses': valid_losses,
               'valid_acc1es': valid_acc1es,
               'valid_acc5es': valid_acc5es,
               'valid_times' : valid_times,
               #'graident_averge': train_conv,
               #'gradient_max': train_maxconv,
               #'gradient_gap': train_minconv,
               #'net_state_dict': net.state_dict(),
               'net_string'  : '{:}'.format(net),
               'finish-train': True
              }
  return info_seed
