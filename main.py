import torch
import tqdm, gc, time
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader
from models.emb_MLPs import *

from dataset import AvazuDataset, Movielens1MDataset, CriteoDataset


def get_dataset(name, path):
    if name == 'movielens1M':
        return Movielens1MDataset(path)
    elif name == 'avazu':
        return AvazuDataset(path)
    elif name == 'criteo':
        return CriteoDataset(path)


def get_model(name,args):
    if name == 'NoSlct': 
        return MLP(args)
    elif name == 'AdaFS_soft': 
        return AdaFS_soft(args)
    elif name == 'AdaFS_hard': 
        return AdaFS_hard(args)


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save({'state_dict': model.state_dict()}, self.save_path) 
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train(model, optimizer, optimizer_model, optimizer_darts, train_data_loader, valid_data_loader, criterion, device, log_interval, controller, darts_frequency):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(train_data_loader, smoothing=0, mininterval=1.0)
    valid_data_loader_iter = iter(valid_data_loader)

    for i, (fields, target) in enumerate(tk0):
        # if model.stage == 1: val_fields.append(fields); val_target.append(target)
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        # for layer_name, param in model.named_parameters(): print('0', layer_name, param[0])

        #Update all params of model if do not use controller
        if not controller:
            optimizer.step()

        #pretrain
        if controller and model.stage == 0:
            optimizer_model.step()

        # search stage, alternatively update main RS network and Darts weights
        if controller and model.stage == 1:
            optimizer_model.step()
            if (i + 1) % darts_frequency == 0:
                # fields, target = torch.cat(val_fields, 0), torch.cat(val_target, 0); val_fields, val_target = [], []
                try:
                    fields, target = next(valid_data_loader_iter)
                except StopIteration:
                    del valid_data_loader_iter
                    gc.collect()
                    valid_data_loader_iter = iter(valid_data_loader)
                    fields, target = next(valid_data_loader_iter)
                fields, target = fields.to(device), target.to(device)
                y = model(fields)
                loss_val = criterion(y, target.float())

                model.zero_grad()
                loss_val.backward()
                optimizer_darts.step()

        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0
        # if i > 100: break



def test(model, data_loader, device):
    model.eval()
    targets, predicts, infer_time  = list(), list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            start = time.time()
            y = model(fields)
            infer_cost = time.time() - start
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
            infer_time.append(infer_cost)
    return roc_auc_score(targets, predicts), log_loss(targets, predicts), sum(infer_time)
    

def main(dataset_name,
         dataset_path,
         model_name,
         args,
         epoch,
         learning_rate,
         learning_rate_darts,
         batch_size,
         darts_frequency,
         weight_decay,
         device,
         pretrain,
         save_dir,
         param_dir):
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length), generator=torch.Generator().manual_seed(42))

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size*2, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    model = get_model(model_name, args ) 
    if pretrain == 0:
        print("trained_mlp_params:",param_dir)
        model.load_state_dict(torch.load(param_dir), strict=False)
    model = model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if model_name != 'NoSlct':
        optimizer_model = torch.optim.Adam(params=[param for name, param in model.named_parameters() if 'controller' not in name], lr=learning_rate, weight_decay=weight_decay)
        optimizer_darts = torch.optim.Adam(params=[param for name, param in model.named_parameters() if 'controller' in name], lr=learning_rate_darts, weight_decay=weight_decay)
    else:
        optimizer_model = None
        optimizer_darts = None

    if pretrain == 1:
        print('\n********************************************* Pretrain *********************************************\n')
        model.stage = 0
        early_stopper = EarlyStopper(num_trials=3, save_path=f'{save_dir}/{model_name}:{dataset_name}_pretrain.pt')
        for epoch_i in range(epoch[0]):
            print('Pretrain epoch:', epoch_i)
            train(model, optimizer, optimizer_model, optimizer_darts, train_data_loader, valid_data_loader, criterion, device, 100, args.controller, darts_frequency)

            auc, logloss,infer_time = test(model, valid_data_loader, device)
            if not early_stopper.is_continuable(model, auc):
                print(f'validation: best auc: {early_stopper.best_accuracy}')
                break
            print('Pretrain epoch:', epoch_i, 'validation: auc:', auc, 'logloss:', logloss)

        auc, logloss,infer_time = test(model, test_data_loader, device)
        print(f'Pretrain test auc: {auc} logloss: {logloss}, infer time:{infer_time}\n')

    print('\n********************************************* Main_train *********************************************\n')
    model.stage = 1
    start_time = time.time()
    if args.controller:
        early_stopper = EarlyStopper(num_trials=3, save_path=f'{save_dir}/{model_name}:{dataset_name}_controller.pt')
    else:
        early_stopper = EarlyStopper(num_trials=3, save_path=f'{save_dir}/{model_name}:{dataset_name}_noController.pt')
    for epoch_i in range(epoch[1]):
        print('epoch:', epoch_i)
        train(model, optimizer, optimizer_model, optimizer_darts, train_data_loader, valid_data_loader, criterion, device, 100, args.controller, darts_frequency)
        auc, logloss,_ = test(model, valid_data_loader, device)
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break
        
        print('epoch:', epoch_i, 'validation: auc:', auc, 'logloss:', logloss)


    auc, logloss, infer_time = test(model, test_data_loader, device)
    print(f'test auc: {auc} logloss: {logloss}\n')
    with open('Record/%s_%s.txt'%(model_name,dataset_name), 'a') as the_file:
        the_file.write('\nModel:%s,Controller:%s,pretrain_type:%s,pretrain_eopch:%s\nDataset:%s,useBN:%s\ntrain Time:%.2f,train Epoches: %d\n test auc:%.8f,logloss:%.8f, darts_frequency:%s\n'
        %(model_name, str(args.controller), str(args.pretrain), str(epoch[0]), dataset_name, str(args.useBN),(time.time()-start_time)/60, epoch_i+1,auc,logloss,str(darts_frequency)))
        if args.model_name == 'AdaFS_hard':
            the_file.write('k:%s, useWeight:%s, reWeight:%s\n'%(str(args.k),str(args.useWeight), str(args.reWeight)))
        if pretrain == 0:
            the_file.write('trained_mlp_params:%s\n'%(str(param_dir)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='movielens1M', help='criteo, avazu, movielens1M')
    parser.add_argument('--model_name', default='AdaFS_soft', help='NoSlct, AdaFS_soft, AdaFS_hard')
    parser.add_argument('--k', type=int, default=0) #选取的特征数,for AdaFS_hard
    parser.add_argument('--useWeight', type=bool, default=True) 
    parser.add_argument('--reWeight', type=bool, default=True)
    parser.add_argument('--useBN', type=bool, default=True)
    parser.add_argument('--mlp_dims', type=int, default=[16,8], help='original=16')
    parser.add_argument('--embed_dim', type=int, default=16, help='original=16')
    parser.add_argument('--epoch', type=int, default=[2,50], nargs='+', help='pretrain/main_train epochs') 
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_rate_darts', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--darts_frequency', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--dropout',type=int, default=0.2)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    parser.add_argument('--add_zero',default=False, help='Whether to add a useless feature')
    parser.add_argument('--controller',default=True, help='True:Use controller in model; False:Do not use controller')
    parser.add_argument('--pretrain',type=int, default=1, help='0:pretrain to converge, 1:pretrain, 2:no pretrain') 
    parser.add_argument('--repeat_experiments', type=int, default=5)
    args = parser.parse_args()

    #对应数据集和与训练模型的路径
    param_dir = args.save_dir
    if args.dataset_name == 'criteo': 
        dataset_path = './dataset/criteo/train.txt'
        param_dir += '/mlp:criteo_noController.pt'
    if args.dataset_name == 'avazu': 
        dataset_path = './dataset/avazu/train'
        param_dir += '/mlp:avazu_noController.pt'
    if args.dataset_name == 'movielens1M': 
        dataset_path = './dataset/ml-1m/train.txt'
        param_dir += '/mlp:movielens1M_noController.pt'

    #对应数据集的field维度
    if args.dataset_name == 'movielens1M':
        args.field_dims = [3706,301,81,6040,21,7,2,3402]
    elif args.dataset_name == 'avazu':
        args.field_dims = [241, 8, 8, 3697, 4614, 25, 5481, 329, 
            31, 381763, 1611748, 6793, 6, 5, 2509, 9, 10, 432, 5, 68, 169, 61]
    elif args.dataset_name == 'criteo':
        args.field_dims = [    49,    101,    126,     45,    223,    118,     84,     76,
           95,      9,     30,     40,     75,   1458,    555, 193949,
       138801,    306,     19,  11970,    634,      4,  42646,   5178,
       192773,   3175,     27,  11422, 181075,     11,   4654,   2032,
            5, 189657,     18,     16,  59697,     86,  45571]
    if args.add_zero:
        args.field_dims.append(0)

    #没有controller的设置
    if args.model_name == 'NoSlct':
        args.controller = False

    #hard selection 没有定义选取特征数k时，赋值fields数的一半
    if (args.model_name == 'AdaFS_soft' or args.model_name == 'AdaFS_hard') and args.controller:
        if args.k == 0:
            args.k = int(len(args.field_dims)/2)
        print(f'\nk = {args.k},\t',
        f'useWeight = {args.useWeight},\t',
        f'reWeight = {args.reWeight}',)
    print(f'\nrepeat_experiments = {args.repeat_experiments}')
    print(f'\ndataset_name = {args.dataset_name},\t',
          f'dataset_path = {dataset_path},\t',
          f'model_name = {args.model_name},\t',
          f'Controller = {args.controller},\t',
          f'useBN = {args.useBN},\t',
          f'mlp_dim = {args.mlp_dims},\t',
          f'epoch = {args.epoch},\t',
          f'learning_rate = {args.learning_rate},\t',
          f'learning_rate_darts = {args.learning_rate_darts},\t',
          f'batch_size = {args.batch_size},\t',
          f'darts_frequency = {args.darts_frequency},\t',
          f'weight_decay = {args.weight_decay},\t',
          f'device = {args.device},\t',
          f'pretrain_type = {args.pretrain},\t',
          f'save_dir = {args.save_dir}\n')
    for i in range(args.repeat_experiments):
        time_start = time.time()
        main(args.dataset_name,
            dataset_path,
            args.model_name,
            args,
            args.epoch,
            args.learning_rate,
            args.learning_rate_darts,
            args.batch_size,
            args.darts_frequency,
            args.weight_decay,
            args.device,
            args.pretrain,
            args.save_dir,
            param_dir)

        print(f'\ndataset_name = {args.dataset_name},\t',
            f'dataset_path = {dataset_path},\t',
            f'model_name = {args.model_name},\t',
            f'Controller = {args.controller},\t',
            f'mlp_dim = {args.mlp_dims},\t',
            f'epoch = {args.epoch},\t',
            f'learning_rate = {args.learning_rate},\t',
            f'learning_rate_darts = {args.learning_rate_darts},\t',
            f'batch_size = {args.batch_size},\t',
            f'darts_frequency = {args.darts_frequency},\t',
            f'weight_decay = {args.weight_decay},\t',
            f'device = {args.device},\t',
            f'pretrain_type = {args.pretrain},\t',
            f'save_dir = {args.save_dir},\t',
            f'training time = {(time.time() - time_start) / 3600}\n')