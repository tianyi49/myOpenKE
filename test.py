import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE, TransR,TransD
from openke.module.model import TransH
from openke.module.model import RESCAL
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

IN_PATH= "/home/tianyi/codeProjects/con_exploit/mydata/Openke_test/after_process_invers_4_26/"



def train_transe():

    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=IN_PATH,
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0)

    # dataloader for test
    test_dataloader = TestDataLoader(IN_PATH, "link", False)

    # define the model
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=100,
        p_norm=1,
        norm_flag=True)

    # define the loss function
    model = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=10.0),
        batch_size=train_dataloader.get_batch_size()
    )

    # train the model
    trainer = Trainer(model=model, data_loader=train_dataloader, train_times=100,alpha=1.0, use_gpu=True)
    trainer.run()
    transe.save_checkpoint(IN_PATH+'checkpoint/transe.ckpt')

    print("#####################TransE Test##########################")
    # test the model
    transe.load_checkpoint(IN_PATH+'checkpoint/transe.ckpt')
    tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)

def   train_transh():
    train_dataloader = TrainDataLoader(
        in_path=IN_PATH,
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0)

    # dataloader for test
    test_dataloader = TestDataLoader(IN_PATH, "link", False)

    # define the model
    transh = TransH(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=100,
        p_norm=1,
        norm_flag=True)

    # define the loss function
    model = NegativeSampling(
        model=transh,
        loss=MarginLoss(margin=10.0),
        batch_size=train_dataloader.get_batch_size()
    )

    # train the model
    trainer = Trainer(model=model, data_loader=train_dataloader, train_times=100, alpha=0.5, use_gpu=True)
    trainer.run()
    transh.save_checkpoint(IN_PATH+'checkpoint/transh.ckpt')
    print("#####################TransH Test##########################")
    # test the model
    transh.load_checkpoint(IN_PATH+'checkpoint/transh.ckpt')
    tester = Tester(model=transh, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)
def train_transr():
    train_dataloader = TrainDataLoader(
        in_path=IN_PATH,
        nbatches=200,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0)

    # dataloader for test
    test_dataloader = TestDataLoader(
        in_path=IN_PATH,
        sampling_mode='link',type_constrain = False)

    # define the model
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=100,
        p_norm=1,
        norm_flag=True)

    model_e = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=5.0),
        batch_size=train_dataloader.get_batch_size())

    transr = TransR(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim_e=100,
        dim_r=100,
        p_norm=1,
        norm_flag=True,
        rand_init=False)

    model_r = NegativeSampling(
        model=transr,
        loss=MarginLoss(margin=10.0),
        batch_size=train_dataloader.get_batch_size()
    )

    # pretrain transe
    trainer = Trainer(model=model_e, data_loader=train_dataloader, train_times=1, alpha=0.5, use_gpu=True)
    trainer.run()
    parameters = transe.get_parameters()
    transe.save_parameters("/home/tianyi/codeProjects/con_exploit/mydata/Openke_test/wo_label_absn_4_26/result/transr_transe.json")

    # train transr
    transr.set_parameters(parameters)
    trainer = Trainer(model=model_r, data_loader=train_dataloader, train_times=100, alpha=1.0, use_gpu=True)
    trainer.run()
    transr.save_checkpoint(IN_PATH+'checkpoint/transr.ckpt')
    print("#####################TransR Test##########################")
    # test the model
    transr.load_checkpoint(IN_PATH+'checkpoint/transr.ckpt')
    tester = Tester(model=transr, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)
def train_rescal():
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=IN_PATH,
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0
    )

    # dataloader for test
    test_dataloader = TestDataLoader(IN_PATH, "link", False)

    # define the model
    rescal = RESCAL(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=100
    )

    # define the loss function
    model = NegativeSampling(
        model=rescal,
        loss=MarginLoss(margin=10.0),
        batch_size=train_dataloader.get_batch_size(),
    )

    # train the model
    trainer = Trainer(model=model, data_loader=train_dataloader, train_times=400, alpha=0.1, use_gpu=True,
                      opt_method="adagrad")
    trainer.run()
    rescal.save_checkpoint(IN_PATH+'checkpoint/rescal.ckpt')
    print("#####################Rescal Test##########################")
    # test the model
    rescal.load_checkpoint(IN_PATH+'checkpoint/rescal.ckpt')
    tester = Tester(model=rescal, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)

def train_transd():
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=IN_PATH,
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0)

    # dataloader for test
    test_dataloader = TestDataLoader(IN_PATH, "link", False)

    # define the model
    transd = TransD(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim_e=200,
        dim_r=200,
        p_norm=1,
        norm_flag=True)

    # define the loss function
    model = NegativeSampling(
        model=transd,
        loss=MarginLoss(margin=10.0),
        batch_size=train_dataloader.get_batch_size()
    )

    # train the model
    trainer = Trainer(model=model, data_loader=train_dataloader, train_times=100, alpha=1.0, use_gpu=True)
    trainer.run()
    transd.save_checkpoint('./checkpoint/transd.ckpt')

    # test the model
    transd.load_checkpoint('./checkpoint/transd.ckpt')
    tester = Tester(model=transd, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)


if __name__ == "__main__":
    train_transe()
    # train_transh()
    # train_rescal()
    # train_transr()
    # train_transd()
    print("#####################All Test Done##########################")






