
import pdb
from pathlib import Path
from runpy import run_path


class metric_selector(object):
    
    def __init__(self, option):
        
        # collect all the metric bank
        metrics = [name for name in Path('src/metric').glob('*') if name.is_dir()]
        metric_names = [name.stem for name in metrics]
        
        # list of metrics to apply
        self.metric_func, self.metric_name = [], []
        for metric_ in option.model.metric_type:
            try:
                idx = metric_names.index(metric_)
            except:
                raise NotImplementedError('wrong metric type : %s' % metric_)
            loaded_file = run_path(str(metrics[idx] / 'logger.py'))
            self.metric_func.append(loaded_file[metric_ + '_Benchmark'](option))
            self.metric_name.append(metric_)
    
    def forward(self, pred, batch, log=True, target_type='disp'):
        result = dict()

        for idx, func in enumerate(self.metric_func):
            out = func.measure(pred, batch, log, target_type)
            result.update({self.metric_name[idx]: out})

        return result
    
    def viewer(self):
        
        for idx, func in enumerate(self.metric_func):
            print('metric_type = %s' % self.metric_name[idx])
            results, t = func.get_value(use_chart=True)
            print(t.draw())