import json
from pathlib import Path
from typing import List
from datetime import datetime
from distribution_inference.config import AttackConfig
from simple_parsing.helpers import Serializable


class Result:
    def __init__(self, path: Path, name: str) -> None:
        self.name = name
        self.path = path
        self.start = datetime.now()
        self.dic = {'name': name, 'start time': str(self.start)}

    def save(self):
        self.save_t = datetime.now()
        self.dic['save time'] = str(self.save_t)
        save_p = self.path.joinpath(self.name)
        self.path.mkdir(parents=True, exist_ok=True)
        with save_p.open('w') as f:
            json.dump(self.dic, f)

    def not_empty_dic(self, dic: dict, key):
        if key not in dic:
            dic[key] = {}

    def load(self):
        raise NotImplementedError("Implement method to model for logger")


class AttackResult(Result):
    def __init__(self, path: Path, name: str, attack_config: AttackConfig):
        super().__init__(path, name)

        def convert_to_dict(dic: dict):
            for k in dic:
                if isinstance(dic[k], Serializable):
                    dic[k] = dic[k].__dict__
                if isinstance(dic[k], dict):
                    convert_to_dict(dic[k])
        self.dic["Attack config"] = attack_config
        convert_to_dict(self.dic)

    def add_results(self, attack: str, prop, vacc, adv_acc=None):
        def check_rec(dic: dict, keys: List):
            if not keys == []:
                k = keys.pop(0)
                self.not_empty_dic(dic, k)
                check_rec(dic[k], keys)
        check_rec(self.dic, ['result', attack, prop])
        if 'adv_acc' in self.dic['result'][attack][prop]:
            self.dic['result'][attack][prop]['adv_acc'].append(adv_acc)
        else:
            self.dic['result'][attack][prop]['adv_acc'] = [adv_acc]
        if 'victim_acc' in self.dic['result'][attack][prop]:
            self.dic['result'][attack][prop]['victim_acc'].append(vacc)
        else:
            self.dic['result'][attack][prop]['victim_acc'] = [vacc]
