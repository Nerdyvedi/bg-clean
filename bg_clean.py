import sys
import os
import yaml
import time
import tensorflow as tf
import tfjs_graph_converter.api as tfjs_api
import tfjs_graph_converter.util as tfjs_util

import numpy as np
import cv2
import datetime
from threading import Thread

from bodypix_functions import calc_padding
from bodypix_functions import scale_and_crop_to_input_tensor_shape
from bodypix_functions import remove_padding_and_resize_back
from bodypix_functions import to_input_resolution_height_and_width
from bodypix_functions import to_mask_tensor
from gf import guided_filter
import sys

def load_config(config_mtime, oldconfig={}):
    """
        Load the config file. This only reads the file,
        when its mtime is changed.
    """

    config = oldconfig
    try:
        config_mtime_new = os.stat("config.yaml").st_mtime
        if config_mtime_new != config_mtime:
            print("Reloading config.")
            config = {}
            with open("config.yaml", "r") as configfile:
                yconfig = yaml.load(configfile, Loader=yaml.SafeLoader)
                for key in yconfig:
                    config[key] = yconfig[key]
            config_mtime = config_mtime_new
    except OSError:
        pass
    return config, config_mtime

def reload_layers(config):
    layers = []
    for layer_filters in config.get("layers", []):
        assert(type(layer_filters) == dict)
        assert(len(layer_filters) == 1)
        layer_type = list(layer_filters.keys())[0]
        layer_filters = layer_filters[layer_type]
        layers.append((layer_type, filters.get_filters(config, layer_filters)))
    return layers


#Global variables    
#cap = cv2.VideoCapture(0)
#cap  = WebcamVideoStream(src=0).start()
#cap  = cv2.VideoCapture("test.mov")
#fps  = FPS().start()
config, config_mtime = load_config(0)

#cap = cv2.VideoCapture(0)




model_path = 'bodypix_resnet_float_050_model-stride16'
#model_path = 'bodypix_resnet_quant2_050_model-stride16'
print("Loading model...")
graph = tfjs_api.load_graph_model(model_path)
print("Done.")

sess = tf.compat.v1.Session(graph=graph)

input_tensor_names = tfjs_util.get_input_tensors(graph)
output_tensor_names = tfjs_util.get_output_tensors(graph)
input_tensor = graph.get_tensor_by_name(input_tensor_names[0])

cap = cv2.VideoCapture(0)
cap2  = cv2.VideoCapture('./video.mp4')

output_stride = 16
multiplier    = 0.2
model_type    = "resnet"


def get_mean():
  frame_count = 0
  bg_frames   = []
  print("Calculating mean background")
  while(True):
    ret,frame   = cap.read()
    frame_count += 1
    if(frame_count>100 and frame_count<200):
      bg_frames.append(frame)
    if(frame_count==200):
      bg_frames  = np.array(bg_frames)
      bg_bw      = bg_frames[:,:,:,1]
      bg_mean    = bg_bw.mean(axis=0) 
      return bg_mean 



def mainloop():
  global masks
  #print("Collecting frames")
  #ret,frame = cap.read()
  # BGR to RGB
  #frames      = collectFrames()
  #print("Frames collected")
  #print("Number of frames : " + str(len(frames)))
  frame_counter = 1
  bg_mean       = get_mean()
  

  while(True):
    #Capture frame

    ret,frame = cap.read()
    ret2,frame2 = cap2.read()
    
    frame_counter += 1
    if frame_counter == cap2.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 1 #Or whatever as long as it is the same as next line
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 1)

    bg   = cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)/255.0
    bg   = cv2.resize(bg,(3840,2160),interpolation=cv2.INTER_AREA)    



    #print(frame.shape)
    if ret:
      input_frame = frame
      #cv2.imshow('portrait segmentation',input_frame)
      #print("Input frame : " + str(input_frame.shape))
      #frame = frame[...,::-1]
      frame = frame.astype(np.float)

      #cv2.imshow('portrait segmentation',input_frame)

      input_height,input_width = frame.shape[:2]
      internal_resolution = 0.2

      target_height,target_width = to_input_resolution_height_and_width(internal_resolution,output_stride,input_height,input_width)
      padT,padB,padL,padR        = calc_padding(frame,target_height,target_width)
      resized_frame              = tf.image.resize_with_pad(frame,target_height,target_width,method=tf.image.ResizeMethod.BILINEAR)
      resized_height,resize_width = resized_frame.shape[:2]

      #Preprocessing for mobilenet
      if(model_type=="mobilenet"):
        resized_frame = np.divide(resized_frame,127.5)
        resized_frame = np.subtract(resized_frame, 1.0)
      
      #Preprocessing for resnet  
      else:
        m = np.array([-123.15, -115.90, -103.06])
        resized_frame =  np.add(resized_frame,m)

      sample_image  = resized_frame[tf.newaxis, ...]
      
      
      
      #print("Resized frame : " + str(sample_image.shape))
      #print("Shape : " + str(sample_image.shape))
      #raise Exception(sample_image.shape)
      results       = sess.run(output_tensor_names,feed_dict = {input_tensor: sample_image})
      #raise Exception(output_tensor_names)
      
      
      #segment_logits = results[6]
      segment_logits = results[6]
      #raise Exception(segment_logits.shape)

             
      scaled_segment_scores = scale_and_crop_to_input_tensor_shape(segment_logits,input_height,input_width,padT,padB,padL,padR,True)
      #raise Exception(scaled_segment_scores.shape)
      upper_threshold = 0.4
      scaled_segment_scores = tf.sigmoid(segment_logits)
      #raise Exception(scaled_segment_scores.shape)

      


      scaled_segment_scores = tf.image.resize_with_pad(scaled_segment_scores,
      input_height, input_width,
      method=tf.image.ResizeMethod.BILINEAR)


      scaled_segment_scores = remove_padding_and_resize_back(scaled_segment_scores,input_height,input_width,padT,padB,padL,padR)

      
      #mask_temp = to_mask_tensor(scaled_segment_scores,0.7)

      

      mask = scaled_segment_scores
      #raise Exception(mask.shape)
      mask = np.reshape(mask,mask.shape[:2])
      m    = mask.copy()
      m[mask>=upper_threshold] = 1.0
      m[mask<upper_threshold] = 0.0
      print(np.sum(m))
      
      # m = m*255.0
      # cv2.imwrite("./mask.jpg",m)
      # raise Exception("Mask written")
      # m[mask<=upper_threshold] = 0.0
      mask = m.copy()
      kernel = np.ones((75,75),np.uint8)
      erode = cv2.erode(m,kernel,iterations = 1)
      dilate= cv2.dilate(m,kernel,iterations=1)
      new_mask = cv2.subtract(dilate,erode)
      #mask.setflags(write=1)
      
      #mask_temp = np.reshape(mask_temp,mask_temp.shape[:2])
      #mask = (mask).astype(np.uint8)
      
      #mask = (mask).view('uint8')[:,:]
      blur_value  = 3
      erode_value = 10 
      # mask[mask>=upper_threshold] = 1.0
      # mask[mask<upper_threshold]  = 0.0


      #mask = cv2.GaussianBlur(mask,(5,5),1)
      #mask  = cv2.GaussianBlur(m,(5,5),1)
      # #Adding erosion
      #mask = cv2.erode(mask,np.ones((erode_value, erode_value),np.uint8), iterations=1)
      # #Adding blur
      #mask = cv2.blur(mask,(blur_value,blur_value))
      # m    = mask.copy()

      # m[mask > upper_threshold] = 1.0
      # m[mask < lower_threshold] = 0.0 
      # mask = m 
      
      #Green channel is considered to be an approximate estimation of Gray image
      foo  = input_frame[:,:,1]
      #raise Exception(mask.shape)
      #dc_gain on the background
      dc_gain_img = np.mean(np.mean(foo[np.where(mask<1.0)]))
      dc_gain_bg  = np.mean(np.mean(bg_mean[np.where(mask<1.0)]))

      # dc_gain_img = np.mean(np.mean(foo[np.where(mask_full<=0.8)]))
      # dc_gain_bg = np.mean(np.mean(mean_img[np.where(mask_full<=0.8)]))


      #Removing extra background
      mask[np.where(  ((mask==1.0) & (np.abs(input_frame[:,:,1] - bg_mean) <=10 + np.abs(dc_gain_bg - dc_gain_img))) & (new_mask>0))] = 0
      #Getting 
      mask[np.where(  ((mask==1.0) & (np.abs(input_frame[:,:,1] - bg_mean) >=15 + np.abs(dc_gain_bg - dc_gain_img))) & (new_mask>0))] = 1
      #mask[(np.where(mask!=1.0)) and (np.where(np.abs(input_frame[:,:,1]-bg_mean) >= 15.0 + np.abs(dc_gain_bg-dc_gain_img)) and np.where(new_mask > 0))] = 1


      #print(bg.shape)
      #print(input_frame.shape)
      mask = mask[:,:,np.newaxis]
      output = (input_frame * mask) + bg * (1 - mask)



      cv2.namedWindow("portrait_segmentation",cv2.WND_PROP_FULLSCREEN)
      cv2.setWindowProperty("portrait_segmentation", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
      cv2.imshow("portrait_segmentation",output)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    #fps.update()
    #out.append(output)

  
  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  """
  while True:
    try:
      mainloop()
    except KeyboardInterrupt:
      print("stopping.")
      break 
  """
  #testBg()
  mainloop()    








