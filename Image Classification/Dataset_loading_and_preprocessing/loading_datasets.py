from importing_librarues.lib import od,os,pd,np,cv2

od.download('https://www.kaggle.com/datasets/alessiocorrado99/animals10')

data_folder_path= "" # path to the dataset folder

folder_names = os.listdir(data_folder_path)

name_map = {
    "cane": "dog",
    "ragno": "spider",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "cavallo": "horse",
}

for folder_name in folder_names:
    if folder_name not in name_map:
        continue

    new_name = name_map[folder_name]

    os.rename(data_folder_path + folder_name, data_folder_path + new_name)

# Loading Image Paths
image_paths = []
categories = []

unique_labels = os.listdir(data_folder_path)[:5]

num_classes = len(unique_labels)

for i, label in enumerate(unique_labels):
    folder_path = data_folder_path + label + "/"

    for imgName in os.listdir(folder_path):
        imgPath = folder_path + imgName

        image_paths.append(imgPath)
        categories.append(i)

df = pd.DataFrame({
    "image_paths": image_paths,
    "categories": categories
})

train_df = pd.DataFrame(columns=['image_paths', 'categories'])

for i in range(num_classes):
    train_df = pd.concat([train_df,df[df['categories'] == i].sample(1000, random_state=42)], axis=0, ignore_index=True)


def centering_image(img):
    size = [256,256]

    img_size = img.shape[:2]

    # centering
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2

    resized = np.zeros(size + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img

    return resized

images = []

for file_path in train_df.image_paths.values:
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*256/img.shape[0]),256)
    else:
        tile_size = (256, int(img.shape[0]*256/img.shape[1]))

    img = centering_image(cv2.resize(img, dsize=tile_size))

    img = img[16:240, 16:240]
    images.append(img)

images = np.array(images)

