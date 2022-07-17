
import pdb
from pathlib import Path
from runpy import run_path


class loss_selector(object):
    
    def __init__(self, option):
        
        # collect all the loss bank
        losses = [name for name in Path('src/loss/depth').glob('*') if name.is_file()]
        losses += [name for name in Path('src/loss/normal').glob('*') if name.is_file()]
        loss_names = [name.stem for name in losses]
        
        # list of loss to apply
        assert(len(option.model.loss_type) == len(option.model.lambdas))
        self.loss_func, self.loss_name, self.lambda_ = [], [], []
        for i, loss_ in enumerate(option.model.loss_type):
            try:
                idx = loss_names.index(loss_)
            except:
                raise NotImplementedError('wrong loss type : %s' % loss_)
            loaded_file = run_path(str(losses[idx]))
            self.loss_func.append(loaded_file[loss_.upper() + "Loss"](option))
            self.loss_name.append(loss_)
            self.lambda_.append(option.model.lambdas[i])
    
    def forward(self, batch, preds):
        result = dict()
        losses = []

        for idx, loss in enumerate(self.loss_func):
            out = loss.forward(batch, preds)
            result.update({self.loss_name[idx] + '_loss': out['loss']})
            if 'abvalue' in out.keys():
                result.update({'abvalue': out['abvalue']})
            losses.append(self.lambda_[idx] * out['loss'])

        result.update({'final_loss': sum(losses)})

        return result