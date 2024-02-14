# Question 1

1. Dataset creation by gathering images from multiple datasets from Kaggle website. It contains two folders of with and without mask folders of images.

2. The without mask dataset from Kaggle is used to generate a masked dataset consisting of cloth, ffp2 and surgical mask. 

3. MaskTheFace is used to convert a face dataset into a masked face dataset. It detects the face from an image and then applies the selected mask to the image.

4. Defining the model and forward pass FaceMaskDetectorCNN model.

5. Performing Batch Normalisation followed by Activation Function.

6. Preparation of data for model using transformations on image.

7. Implementation of Stratified KFold.

8. Used Dataloader helper function for automatic batching.

9. Configuration of Adam optimiser.

10. Training accuracy and loss are calculated after each epoch.

11. Evaluation is done using Stratified KFold and the confusion matrix is visualized. 

# Question 2
Dataset description:
The dataset used in this project is an amalgamation of images taken from multiple datasets from kaggle. Kaggle dataset contains two folders with mask and without mask. The without mask dataset from kaggle and some random images from kaggle, are used to generate a masked dataset consisting of cloth, ffp2 and surgical mask. Dataset is a balanced dataset that consists of 5985 RGB images in 4 folders named as cloth, surgical, ffp2 and no (without mask). 

| Type | Number of images |
|:----------------------|:-----:|
| Cloth Mask | 1493 |
| FFP2 Mask | 1499 |
| Surgical Mask | 1498 |
| Without Mask | 1495 |
| Total | 5985 |

MaskTheFace is used to add masks to the without mask images. It works as -
- opensource face landmark detection
- estimation of mask position 
- selecting right template based on face tilt
- warp the mask and overlay it on image with adjusted brightness.
So basically, we input a image, then select the type of mask, and that mask gets coated on our image. So I created my dataset with the help of this and made 4 seperate folders with distinct masks' type.

Link to the datasets used :
[1] https://www.kaggle.com/omkargurav/face-mask-dataset
[2]https://www.kaggle.com/bmarcos/image-recognition-gender-detectioninceptionv3/data
Link to MaskTheFace repository: https://github.com/aqeelanwar/MaskTheFace

Images are of the form of input of [3*32*32], where 3 is the number of channels i.e. RGB in our case and 32, 32 are the height and width of the image. 

As for AUC, I tried to implement the following steps as guided by TA, but was not able to implement it successfuly.
- Loading of dataset into a single numpy array. Then convert it into binary classification by renaming the labels != 0 to 1.
- Images to be converted to grayscale, by diving by 3 or taking mean
- Resizing of images in 4x4 as my image size was originally 32x32, so it divided into 8 regions.
- Converting this array into pandas dataframe
- Naming of columns for all 8 regions 
- Calculating AUC using sklearn function

I tried to create a function consisting of above steps but I was getting errors in resizing, so I was not able to include that code in file.


# Question 3

1,2,3. 
For these steps relate to creating a dataframe for dataset and appending each type of folder images in it. 
datasetPath = Path('dataset')
noMaskPath = datasetPath/'no'
clothMaskPath = datasetPath/'cloth'
FFP2MaskPath = datasetPath/'FFP2'
surgicalMaskPath = datasetPath/'surgical'
maskDF = pd.DataFrame()

For example for without mask images I used the below code, and proceeded similary for all of them.

for imgPath in tqdm(list(noMaskPath.iterdir()), desc='no'):
    maskDF = maskDF.append({
        'image': str(imgPath),
        'mask': 0
    }, ignore_index=True)

4, 5. 
A CNN architecture with 4 convolution layers is used to extract features from the images. 
Each of the 4 convolution layers comes with a kernel size of (3,3), stride of (1,1) , and a padding of (1,1). The convolution layer is followed by a Batch Norm layer which is implemented using nn.BatchNorm2d(). Batch normalization is used to distribute data uniformly across a mean that is best to the network, before passing it to the activation function in order to avoid undershoot or overshoot.

Here, LeakyReLU activation function is used. It overcomes the dying ReLU problem. There are two pooling layers , each after every 2 convolution layers. The pooling used is max pooling with a filter of (2,2) and a stride of 2. Padding allows to use the CONV layer without shrinking the height and width of the 
image. It also helps to retain more information at the border of the image. 

Dropout is used to prevent the model from overfitting. Here it is placed on fully connected layers as these layers have a high number of parameters and are more likely to overfit. The dropout parameter is set to 0.1.

Finally, the output from the final pooling layer is flattened and fed as an input to the fully connected layer. Neurons in a fully connected layer are connected to all the activations in the previous layer. Therefore, it helps to classify the data into multiple classes.

Summary of model can be viewed in output file model.npy

6, 7.
A pickle file is loaded that stored the created dataset. Images in the dataset are transformed using torchvision.transforms. They are resize to (32, 32) , normalized and converted to tensor. StratifiedKFold is used for the train test split.
Here is the code extract for it - 
def prepare_data(mask_df_path) -> None:
        mask_df = pd.read_pickle(mask_df_path)
        print(mask_df['mask'].value_counts())
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        train_folds = []
        validate_folds = []
        for train_index, validate_index in skf.split(mask_df, mask_df['mask']):
            train_folds.append(MaskDetectionDataset(mask_df.iloc[train_index]))
            validate_folds.append(MaskDetectionDataset(mask_df.iloc[validate_index]))
        return [
            train_folds, validate_folds,CrossEntropyLoss()
            ]

8.A Dataloader helper function is created for automatic batching. Dataloaders are defined for training and validation. The model is trained with a batch size of 32. The shuffle attribute is made true so that the model can train better as it receives data that is not in a repetitive trend.

code extract - 
def train_dataloader(train_df) -> DataLoader:
    return DataLoader(train_df, batch_size=32, shuffle=True, num_workers=0)

def val_dataloader(validate_df) -> DataLoader:
    return DataLoader(validate_df, batch_size=32, num_workers=0) 

9, 10.
The optimizer used here is Adam optimizer and the learning rate is set to 0.001.

The number of epochs are 10 and for each epoch the data is loaded using Dataloader. The model is trained using the face_mask_detector_cnn() model. The loss is calculated using cross entropy loss. The optimizer.zero_grad() function sets the gradients to zero. The backward() function is called on the loss to calculate back propagation. The optimizer.step() iterates over all the parameters and updates them using 
their internally stored grad values. Training accuracy and loss are calculated after each epoch(but not printed).

Code - 
def train_model(train_fold):
    acc_list = []
    loss_list = []
    optimizer = Adam(face_mask_detector_cnn.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        total=0
        correct=0
        loss_train = 0.0
        for i, data in enumerate(train_dataloader(train_fold), 0):
            inputs, labels = data['image'], data['mask']
            labels = labels.flatten()
            outputs = face_mask_detector_cnn(inputs)
            loss = cross_entropy_loss(outputs, labels)
            loss_list.append(loss.item())
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            #training accuracy
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1) 
            correct += (predicted == labels).sum().item() 
            loss_train += loss

11.Model is evaluated using a stratified k fold strategy on training and testing set. The model is iterated over 10 folds with 10 epochs each and a batch size of 32. The average accuracy, precision, recall and f1-score of these folds are calculated at the end. The confusion matrix is visualized with plot_cm() 
function. From figs folder, confusion matrix can be viewed saved as matrix.png

Metrics      Score
accuracy     0.848126
precision    0.852032
recall       0.848160
f-score      0.848385


From the confusion matrix and the metrics scores the following observations 
are noted:
1. Add more convolution layers to extract more features
2. Increase the accuracy of the model
3. Get more diverse data such as people of different age groups and different races and increase the amount of dataset
4. The model does not predict well for surgical mask. 

Other Referrences include - 
https://github.com/aqeelanwar/MaskTheFace/blob/master/images/block_diag.png
BolognaNN - Photo Geolocation in Bologna with Convolutional Neural Networks - Scientific Figure on ResearchGate. 
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
https://towardsdatascience.com/convolutional-neuranetwork-17fb77e76c05
https://towardsdatascience.com/forward-and-backward-propagationsfor-2d-convolutional-layers-ed970f8bf602
https://towardsdatascience.com/the-dying-relu-problem-clearlyexplained-42d0c54e0d24
https://deeplizard.com/learn/video/0LhiS6yu2q