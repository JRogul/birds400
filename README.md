# birds400
classification of 400 birds species
## models
Models were chosem fron available torchvision models for computer vision tasks. All 3 models were downloaded as pretrained. 
Then, last layers were removed, all layers were frozen while training except the replaced final layers.
Then, all models were trained for 40 epochs, without usage of any learning scheduler which could potentially increase performance of all models.
The ensembling was created simply by taking mode of prediction. Final results are saved in final_results.csv.
