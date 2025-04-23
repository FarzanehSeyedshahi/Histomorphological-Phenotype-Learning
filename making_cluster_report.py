from PIL import Image
from reportlab.lib.pagesizes import letter, A2
from reportlab.pdfgen import canvas
import pandas as pd
import os
from reportlab.lib.utils import ImageReader

from tqdm import tqdm

import PIL
PIL.Image.MAX_IMAGE_PIXELS = 320000000

# Function to create PDF from images with annotations
def create_pdf(images_folder, annotations_csv, output_pdf):
    annotations_data = pd.read_csv(annotations_csv)
    page_size = A2
    c = canvas.Canvas(output_pdf, pagesize=page_size)
    width, height = page_size 
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = sorted(image_files, key=lambda x: int(x.split('_')[1]))
    for image_file in tqdm(image_files):
        image_path = os.path.join(images_folder, image_file)
        image = Image.open(image_path)
        img_width, img_height = image.size
        cropped_image = image.crop((15500, 1000, img_width-15750, img_height-500))
        img_width, img_height = cropped_image.size
        resized_image = cropped_image.resize((int(img_width/8), int(img_height/8)), Image.LANCZOS)
        img_width, img_height = resized_image.size
        hpc = image_file.split('_')[1]
        rl_image = ImageReader(resized_image)
        
        c.setFont('Times-Roman', size=25)
        c.drawCentredString(width/2, height-45 , 'HPC Number: '+ hpc)
        c.setFont('Times-Roman', size=14)
        c.drawImage(rl_image,45,height-img_height-60 , width= img_width, height = img_height, mask='auto')

        image_annotations = annotations_data[annotations_data['HPC'] == int(hpc)]
        for i, annotation in image_annotations.iterrows():
            c.rect(45, height-img_height-120, img_width, 90, fill=0)
            c.drawString(60, height-img_height-60 , 'Annotation: '+ annotation['Summary'])
            # c.drawString(60, height-img_height-80 , 'Category: '+annotation['category'])
            c.drawString(60, height-img_height-100 , 'Main Pattern: '+annotation['main_pattern'])
        c.showPage()
    c.save()


from data_manipulation.data import Data
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_hovernet_annotations(cluster_folder, groupby='leiden_2.0', fold=0):
    csv_list = os.path.join(cluster_folder, '%s_fold%s' % (groupby.replace('.', 'p'), fold), 'backtrack')
    sorted_csv_list = sorted(os.listdir(csv_list), key=lambda x: int(x.split('_')[1]))
    dataset = 'Meso_250_subsampled'
    data = Data(dataset=dataset, marker='he', patch_h='224', patch_w='224', n_channels='3', batch_size=64, project_path="/raid/users/farzaneh/Histomorphological-Phenotype-Learning/", load=True)
    img_folder = os.path.join(cluster_folder, '%s_fold%s' % (groupby.replace('.', 'p'), fold), 'img_folder')
    os.makedirs(img_folder, exist_ok=True)
    for csv_file in sorted_csv_list:
        df = pd.read_csv(os.path.join(csv_list, csv_file))
        csv_img_folder = os.path.join(img_folder, csv_file.split('.')[0])
        os.makedirs(csv_img_folder, exist_ok=True)
        for i, row in tqdm(df.iterrows()):
            img = data.training.images[int(row['indexes'])]/255.
            plt.imsave(os.path.join(csv_img_folder , 'image_%s.png' % row['indexes']), img)
        break

 
def add_forest_plots_to_pdf(forest_plot_path, output_pdf):
    # add one page forest plot to pdf

    page_size = A2
    c = canvas.Canvas(output_pdf, pagesize=page_size)
    width, height = page_size 
    image = Image.open(forest_plot_path)
    img_width, img_height = image.size
    resized_image = image.resize((int(img_width/2), int(img_height/2)), Image.LANCZOS)
    img_width, img_height = resized_image.size
    rl_image = ImageReader(resized_image)
    c.drawImage(rl_image,45,height-img_height-60 , width= img_width, height = img_height, mask='auto')
    c.showPage()
    c.save()



if __name__ == "__main__":
    main_path = '/nfs/home/users/fshahi/Projects/Histomorphological-Phenotype-Learning/'

    resolution = '2p0'
    meta_folder = '750K'
    fold = 4
    dataset = 'acmeso'


    images_folder = "{}/results/BarlowTwins_3/{}/h224_w224_n3_zdim128/{}/leiden_{}_fold{}/images/".format(main_path, dataset, meta_folder, resolution, fold)
    annotations_csv = "{}/files/meso_annotations_{}.csv".format(main_path, meta_folder)
    output_pdf = "{}/results/BarlowTwins_3/{}/h224_w224_n3_zdim128/{}/leiden_{}_fold{}/cluster_report.pdf".format(main_path, dataset, meta_folder, resolution, fold)
    cluster_folder = "{}/results/BarlowTwins_3/{}/h224_w224_n3_zdim128/{}/".format(main_path, dataset, meta_folder)
    # h5_file = "{}/results/BarlowTwins_3/{}/h224_w224_n3_zdim128/{}/hdf5_{}_he_complete_metadata_filtered.h5".format(main_path, dataset, meta_folder, dataset)
    # forest_plot_path = "{}results/BarlowTwins_3/Meso_250_subsampled/h224_w224_n3_zdim128/meso_nn250/leiden_2p0_fold0/forest_plot.png".format(main_path)
    create_pdf(images_folder, annotations_csv, output_pdf)
    # get_hovernet_annotations(cluster_folder, groupby='leiden_2.0', fold=0)
    # add survival information to pdf
    # add_forest_plots_to_pdf(forest_plot_path, output_pdf)
# 