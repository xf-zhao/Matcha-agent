from sound_module import SoundClassifier

sc = SoundClassifier(data_path='resources', 
                     ignore=['fibre',])
sc.offline_train()
sc.offline_test()