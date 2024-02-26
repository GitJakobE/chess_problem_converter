from model.pytorch_model import TouchDataModel

def test_train_pytorch() ->None:
    dm = TouchDataModel()
    dm.init_torch_model()
    dm.train_torch_model(training_lib="../../trainingset")
    dm.save_model()

def test_load_data_model() ->None:
    dm = TouchDataModel()
    dm.load_model()
