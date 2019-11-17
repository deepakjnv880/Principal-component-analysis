import cv2 as cv
import numpy as np
import sys
from PIL import Image
import os
import random
# np.set_printoptions(threshold=sys.maxsize) # for showing full numpy array without truncation
# np.set_printoptions(suppress=True)# for suppreesing sciecmtific printing
train_dir='training/'
testing_dir='testing/'
result_dir='result/'


def reconstruction(image_vector,eigen_vectors,score,mean_vector,fname):
    print("i am in reconstrdcutoin")
    print("==",np.shape(eigen_vectors))
    print("** ",np.shape(score))
    recon = np.dot((eigen_vectors), score)
    # recon = np.dot(np.linalg.inv(eigen_vectors.T), score)
    print("=================")
    for i in range(len(recon)):
        recon[i]+=mean_vector[i];
    recon = np.uint8(np.clip(recon,0,255))
    # recon=np.clip(recon,0,255)

    # print(recon)
    print("end")
    print(recon)
    get_image(recon,fname)

def convert_all_image_to_vector(digit):
    file_names=[]
    all_image_vector=[]
    number_of_image=0
    for fname in os.listdir(train_dir+str(digit)):
        # print(fname)
        file_names.append(fname)
        image=cv.imread(train_dir+str(digit)+str("/")+fname)
        image=cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image=image.flatten()
        all_image_vector.append(image)
        number_of_image+=1
    return number_of_image,all_image_vector,file_names

def get_image(reconstructed_image_vector,fname):
    image_matrix=np.reshape(reconstructed_image_vector,(28, 28))
    img = Image.fromarray(image_matrix, 'L')
    img.save(result_dir+fname)


def add_noise(all_image_vector, level):
    for i in range(len(all_image_vector)):
        noise = np.random.normal(0, 10, 28*28)
        all_image_vector[i] = np.add(all_image_vector[i], float(level)*noise)
    return all_image_vector

def main():
    print("In how many dimension you want to reduce your feature vector :")
    number_of_principal_component=int(input())
    if number_of_principal_component<1 or number_of_principal_component>28:
        print("Please enter valid number of principal component.")
        return
    # number_of_principal_component=28
    print("Enter the digit you want to analize :")
    digit=int(input())
    # digit=9
    number_of_image,all_image_vector,file_names=convert_all_image_to_vector(digit)
    all_image_vector=np.asarray(all_image_vector)
    print("Do you want to add noise(y or n) :")
    check=(input())
    if check=="y":
        all_image_vector = add_noise(all_image_vector, 0.2)
    mean_vector=np.mean(all_image_vector, axis=0)
    # for x in all_image_vector:
    #     for i in range(len(mean_vector)):
    #         x[i]=x[i]-mean_vector[i]

    convarience_matrix=np.cov(np.array(all_image_vector).T)
    eig_vals, eig_vecs = np.linalg.eig(np.array(convarience_matrix))
    eig_vals=np.real(eig_vals)
    eig_vecs=np.real(eig_vecs)
    print("Eigenvector and Eigenvalue shapes are")
    print(eig_vecs.shape, eig_vals.shape)

    temp = []
    for i in range(len(eig_vals)):
        temp.append([eig_vals[i],eig_vecs[:,i]])

    temp.sort(key = lambda x:x[0], reverse=True)

    eig_vecs = []
    for i in range(number_of_principal_component*number_of_principal_component):
        eig_vecs.append(temp[i][1])

    eig_vecs = np.asarray(eig_vecs)
    print(eig_vecs)
    ### second method ##########################
    ##return the sorted index of eigen value in ascending order
    # sorted_index = np.argsort(eig_vals,kind='quicksort')
	# # reverse the sorted index sothat it become in descending order
    # sorted_index = sorted_index[::-1]
	# # arrange the eigen vector as per eigen value
    # eig_vecs = eig_vecs[sorted_index]
    #
    # # taking number_of_principal_component eigen vecs
    # eig_vecs = eig_vecs[:, range(number_of_principal_component*number_of_principal_component)]
    for file in os.listdir(result_dir):
        if file.endswith('.jpg'):
            os.remove(result_dir+"/"+file)
    for i in os.listdir(testing_dir+str(digit)):
        image=cv.imread(testing_dir+str(digit)+"/"+i)
        image=cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image=image.flatten()
        # print("################################################################")
        score = np.dot(eig_vecs, np.array(image))
        reconstruction(image,np.transpose(eig_vecs),score,mean_vector,i)

if __name__ == '__main__':
    main()
