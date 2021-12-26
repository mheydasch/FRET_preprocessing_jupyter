#import blend_modes
import cv2
import gc
import glob
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pickle
import tensorflow as tf
import time
from collections import deque
from scipy import ndimage as nd
from skimage import exposure
from skimage import io
from skimage import io
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential, Model

#%%

#https://github.com/87surendra/Random-Forest-Image-Classification-using-Python/blob/master/Random-Forest-Image-Classification-using-Python.ipynb
#https://github.com/bnsreenu/python_for_microscopists/blob/master/062-066-ML_06_04_TRAIN_ML_segmentation_All_filters_RForest.py


def annotations_to_tensor(feature_matrix,mask):
    #feature matrix dim: [x,y,nb_features]
    #possible mask elements: NaN: not annotated, int[0,1]: class annotation
    y_labels=[] #where class labels are stored
    X_features=[] #where feature vectors are stored
    for x,y in np.argwhere(~np.isnan(mask)):
        y_labels.append(mask[x,y])
        X_features.append(feature_matrix[x,y,:])
    #turn list into np array
    X_features = np.asarray(X_features)
    return X_features,y_labels



def apply_clf(args):
    path, clf = args[0],args[1]
    frames = np.load(path)
    output_stack = []
    
    
    shapes = [(1024,1024),(512,512),(256,256),(128,128)]
    models = init_VGG16_pyramid(shapes)
    
    for frame_nb in (range(np.shape(frames)[0])):
        frame = frames[frame_nb,:,:]
        features = fd_VGG16_pyramid(frame,models,shapes)
        to_predict = features.reshape(np.shape(features)[0]*np.shape(features)[1],np.shape(features)[2])
        
        prediction_vector = clf.predict_proba(to_predict)[:,0] #[:,] to only use the prob. of the first class
        prediction_image = np.reshape(prediction_vector, (540, 540))
        output_stack.append(prediction_image)
        
        feature_image = []
        feature_vector = []
        prediction_vector = []
        prediction_image = []
        
    output_stack = np.array(output_stack)
    output = output_stack*255
    output = output.astype('uint8')
    
    return output



def apply_semantic_segmentation(chunk_paths, model):
    arg_stack = []

    #Create pool before loading dataset to avoid memory leaks
    pool = multiprocessing.Pool(processes= len(chunk_paths))
    time_start = time.time()
    for chunk_path in chunk_paths:
        arg_stack.append([chunk_path,model])

    print("Start workers ...")
    #delete the tiff file from memory as child processes will have copy of memory of parent:
    #https://stackoverflow.com/questions/49429368/how-to-solve-memory-issues-problems-while-multiprocessing-using-pool-map
    #tiff_orig = []


    results = pool.imap(apply_clf, arg_stack, chunksize = 1)
    pool.close() # No more work
    print("Wait for completion ...")
    pool.join()  # Wait for completion

    # Extract the results from the iterator object and store as tiff file.
    # The objects are automatically removed from the iterator once they are parsed - don't forget to store them!
    stack = []
    for result in results:
        stack.append(result)
    stack_array = np.array(stack)
    stack_array = np.concatenate(stack_array,axis=0)
    stack_array = stack_array.squeeze()
    
    nb_frames = np.shape(stack_array)[0]
    time_stop = time.time()
    time_total =  time_stop - time_start
    print("Semantic segmentation done. Processing time per frame: "+str(round(time_total/nb_frames, 2) )+" seconds. Total time: "+str(round(time_total/60,2))+" minutes.") 
    return stack_array


def fd_VGG16_scaled(img,model,shape = (540, 540)):
    
    #resize image to new shape  
    input_image_stacked = np.expand_dims(img, axis=-1)
    if shape != (540, 540):
        img = tf.image.resize(input_image_stacked,shape)
    #new_model.summary()
    #as it works only with 3 input channels: stack nuclear channel
    stacked_img = np.stack((img,)*3, axis=2)
    stacked_img = np.squeeze(stacked_img)
    stacked_img = stacked_img.astype(np.float32)
    
    stacked_img = stacked_img.reshape(-1, shape[0], shape[1], 3)
    
    #predict class in keras for each pixel
    features=model.predict(stacked_img)
    
    #remove extra dim
    fv_VGG16= np.squeeze(features)
    
    #scale up to match original img size
    #fv_VGG16 = resize(fv_VGG16,(1024,1024))
    if shape!= (540, 540):
        fv_VGG16 = tf.image.resize(fv_VGG16,(540,540))
    return fv_VGG16


def fd_VGG16_pyramid(img,models,shapes):
    #img - input image to calculate vgg response of
    #models - list of all vgg16 models 
    #shapes - corresponding shapes
    
    fv_list = []
    for model,shape in zip(models,shapes):
        fv = fd_VGG16_scaled(img,model,shape)
        fv_list.append(fv)
    
    global_feature = np.concatenate(fv_list,axis=2)
    return global_feature    
    
    

def init_VGG16_pyramid(input_shapes=[(540, 540)]):
    models = []
    for shape in input_shapes:
        keras_shape = (shape[0],shape[1],3) #add color channel
        VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=keras_shape)
        #VGG_model.summary()

        #disable training (use pretrained weights)
        for layer in VGG_model.layers:
            layer.trainable = False

        #only use up to last layer where input size is still 1024x1024
        new_model = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv2').output)
        models.append(new_model)
    return models



def render_output(show_annotations, show_prediction, blending_function, blending_alpha, background_img, prediction_img, annotations):

    
    if prediction_img == [] or not show_prediction:
        output_img = background_img
    else:
        dummy_alpha = np.full_like(prediction_img,255)
        dummy_alpha = np.expand_dims(dummy_alpha, axis = -1)

        #convert p-map from [0-1] to cmap 
        prediction_img = (prediction_img*255).astype('uint8')
        prediction_img = cv2.applyColorMap(prediction_img, cv2.COLORMAP_JET)
        background_img = background_img.astype(float)
        prediction_img = prediction_img.astype(float)
        background_img = np.concatenate((background_img, dummy_alpha),  axis = -1)
        prediction_img = np.concatenate((prediction_img, dummy_alpha),  axis = -1)

        #blend the bg image and prediction img
        output_img = blending_function(background_img,prediction_img,blending_alpha)
        output_img = np.uint8(output_img)
        output_img = output_img[:,:,0:3] #remove alpha channel 
            
        
    #add annotations
    if show_annotations:
        col_bg = [255,100,100] #color for background annotations
        col_fg = [100,255,100] #color for foreground annotations
        is_bg = np.where(annotations == 0)
        is_fg = np.where(annotations == 1)
        #https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/AccessingDataAlongMultipleDimensions.html#Supplying-Fewer-Indices-Than-Dimensions
        output_img[is_bg]=col_bg
        output_img[is_fg]=col_fg
        
    cv2.imshow('image',output_img)
    return output_img


def interface(input_img, classifier, blending_alpha = 0.5, mask = None):
    drawing = False # true if mouse is pressed
    mode = True
    cv2.destroyAllWindows() #close any windows left 
    
    #Adapt the histogram (rescale brightness)
    p2, p98 = np.percentile(input_img, (2, 98))
    img8 = exposure.rescale_intensity(input_img, in_range=(p2, p98))
    
    #Scale to range [0-255] and conv to uint8 from 16bit
    img8 = (img8-np.amin(img8))/(np.amax(img8)-np.amin(img8))
    img8 = img8*255
    img8 = np.round(img8)
    img8 = img8.astype('uint8')
    
    #create 3channel grayscale image
    background = np.stack((img8,)*3, axis=-1)
    prediction_img = [] #init as empty
    
    shapes = [(1024,1024),(512,512),(256,256),(128,128)]
    models = init_VGG16_pyramid(shapes)
    features = fd_VGG16_pyramid(input_img,models,shapes)
    
    radius = 2
    
    #all available blendmodes
    #blend_mode_names = ['soft_light','lighten_only','dodge','addition','darken_only','multiply','hard_light','difference','subtract','grain_extract','grain_merge','divide','overlay','normal'] 
    
    #possibly useful blendmodes:
    blend_mode_names =['soft_light','lighten_only','multiply','grain_extract','overlay','normal'] 
    
    blend_mode_selected = 0 #default blend mode
    blending_function = getattr(blend_modes, blend_mode_names[blend_mode_selected])

    show_annotations = True
    show_prediction = True
    
    #empty annotation_layer
    annotation_stack_max = 100 #max number of undo steps
    annotation_empty = np.zeros(np.shape(background)[:2])
    annotation_empty.fill(np.nan)
    
    #init stack for undo functionality
    annotation_stack = deque()
    annotation_stack.append(annotation_empty)
    
    def draw_circle(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            if mode == True:                
                #cv2.circle(output,(x,y),radius,(255,100,100),-1)
                
                #get last annotation and update it, append it to stack
                annotation_last = annotation_stack[-1].copy() #peek at rightmost item                  
                cv2.circle(annotation_last,(x,y),radius,(0),-1)
                annotation_stack.append(annotation_last)
            else:
                #cv2.circle(output,(x,y),radius,(100,255,100),-1)
                annotation_last = annotation_stack[-1].copy()     
                cv2.circle(annotation_last,(x,y),radius,(1),-1)
                annotation_stack.append(annotation_last)
                
            #if max number of undo steps reached, delete oldest step
            if len(annotation_stack) == annotation_stack_max:
                annotation_stack.popleft()
        
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    cv2.imshow('image',background)
    output = background.astype('uint8')
    dest = np.empty(np.shape(background),np.uint8)
    dest = dest.astype('uint8')
    update = False
    
    
    ## TODO replace this with previous features/annotations instead of a previous mask
    if type(mask) != type(None):
        no_mask_initialized = False
        mask_one_channel = mask
        
    else:
        no_mask_initialized = True
        mask = np.empty(np.shape(background), np.uint8)
        mask_one_channel = np.zeros(np.shape(background)[:2])
        mask_one_channel.fill(np.nan)
        output = background.copy()
        

    no_prediction_initialized = True
    
    ## MAIN LOOP ##
    while(1):
        
        annotations = annotation_stack[-1] #peek at latest annotation

        ## Update Classifier
        if update:
            print('\r' + '[TRAINING CLASSIFIER]          ', end='')
            
            # extract the annotated pixels
            X,y = annotations_to_tensor(features,annotations)
            classifier.fit(X, y)
            to_predict = features.reshape(np.shape(features)[0]*np.shape(features)[1],np.shape(features)[2])
            prediction_list = classifier.predict_proba(to_predict)[:,0] #use [:,0] if displaying probab. 
            prediction_img = np.reshape(prediction_list, (540, 540))
            update = False


            no_prediction_initialized = False
          
        if mode:
            print('\r' + 'Click to label [BACKGROUND]', end='')
        else: 
            print('\r' + 'Click to label [  NUCLEI  ]', end='')
                
        ## update the output display
        render_output(show_annotations, show_prediction, blending_function, blending_alpha, background.copy(), prediction_img, annotations)
        
                
        k = cv2.waitKey(1) & 0xFF
        
        
        ## KEY BINDINGS
        if k == ord('m'):
            mode = not mode
            if mode:
                print('\r' + 'Click to label [BACKGROUND]', end='')
            else: 
                print('\r' + 'Click to label [  NUCLEI  ]', end='')
                
        elif k == ord('q'):
            break
            
        elif k == ord('u'):
            update = True   
            
        elif k in list(map(ord,list(map(str,range(0,10))))):
            #change radius of pen to number key clicked
            radius = int(chr(k))    
            
        elif k == ord('z'):
            #undo function: if more than 1 elem left, go back
            if len(annotation_stack) > 1:
                annotation_stack.pop()
        
        if k == ord('a'):
            #show / hide annotations
            show_annotations = not show_annotations
            
        if k == ord('p'):
            #show / hide annotations
            show_prediction = not show_prediction
        
        elif k == ord('b'):
            #change blend mode
            blend_mode_selected = (blend_mode_selected + 1) % len(blend_mode_names)
            blending_function = getattr(blend_modes, blend_mode_names[blend_mode_selected])
            
    cv2.destroyAllWindows()
    labels = mask_one_channel 
    return classifier #return classifier