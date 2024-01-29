
import os
import SimpleITK as sitk
import numpy as np
import time

def get_largest_connected_component(image, label_value, log = None):
    binary_image = sitk.BinaryThreshold(image, lowerThreshold=label_value, upperThreshold=label_value)
    
    cc_filter = sitk.ConnectedComponentImageFilter()
    labeled_image = cc_filter.Execute(binary_image)
    
    #print(f'label ({label_value})의 component 개수: ', cc_filter.GetObjectCount())
    if not log == None:
        log.write(f'    label ({label_value})의 component 개수: {cc_filter.GetObjectCount()}\n')
    
    label_sizes = sitk.LabelShapeStatisticsImageFilter()
    label_sizes.Execute(labeled_image)
    
    largest_label = 0
    largest_size = 0
    for label in label_sizes.GetLabels():
        #print(f'{label}번 Component의 voxel개수: ', label_sizes.GetNumberOfPixels(label))
        if not log == None:
            log.write(f'        {label}번 Component의 voxel개수: {label_sizes.GetNumberOfPixels(label)}\n')
        if label_sizes.GetNumberOfPixels(label) > largest_size:
            
            largest_label = label
            largest_size = label_sizes.GetNumberOfPixels(label)
    
    result_image = sitk.BinaryThreshold(labeled_image, lowerThreshold=largest_label, upperThreshold = largest_label, insideValue=label_value, outsideValue=0)
    
    return result_image
    

def main():
    mod = '.nii.gz'
    #mod = '.nrrd'
    
    if mod=='.nii.gz':
        #labelFolder = os.path.normpath(r'C:\Users\Nature\Desktop\Dataset\_TTE_images\label')
        labelFolder = os.path.normpath(r'C:\Users\Nature\Desktop\Dataset\_TTE_images_cat\train\label\nii.gz')
        assert os.path.isdir(labelFolder), '유효하지 않은 경로'
        
        label_list = [os.path.join(labelFolder, file) for file in os.listdir(labelFolder)]
        label_list = [file for file in label_list if os.path.splitext(file)[1] in ('.nii', '.gz')]  
    elif mod == '.nrrd':
        labelFolder = os.path.normpath(r'C:\Users\Nature\Desktop\nrrd')
        assert os.path.isdir(labelFolder), '유효하지 않은 경로'
        
        label_list = [os.path.join(labelFolder, file) for file in os.listdir(labelFolder)]
        label_list = [file for file in label_list if os.path.splitext(file)[1] == '.nrrd' and 'intf_lab_gen_'in file and file.endswith('.nrrd')]
        
    saveFolder = os.path.normpath(r'C:\Users\Nature\Desktop\nrrd')
    
    amount = len(label_list)
    assert amount>0, f'폴더에 nii.gz 파일이 없음\nlabel:{amount} 개'
    
    label_value = [1,2]
    
    with open('log.txt','w') as log:
    #라벨 로드
        for i, path in enumerate(label_list):

            log.write(f'{i}번째 GT 이미지\n')
            label = sitk.ReadImage(path)
            result_images = [get_largest_connected_component(label,label_value, log) for label_value in label_value]
            #merged_image = result_images[0] + result_images[1]
            
            #if mod == '.nii.gz':
                #sitk.WriteImage(merged_image, os.path.join(saveFolder,f'label_{i:03d}.nii.gz'))
            #elif mod == '.nrrd':
                #sitk.WriteImage(merged_image,os.path.join(saveFolder,f'pp_mask_{i:02d}.nrrd') )

if __name__ == '__main__':
    main()