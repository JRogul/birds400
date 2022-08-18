# birds400
classification of 400 birds species
## models
Models were chosem fron available torchvision models for computer vision tasks. All 3 models were downloaded as pretrained. 
Then, last layers were removed, all layers were froze except the replaced final layers.
Then, all models were trained for 40 epochs, without sage of any learning scheduler which could potentially increase performance of all models.
