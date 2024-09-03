from importing_librarues.lib import plt,sns
from Dataset_loading_and_preprocessing.loading_datasets import unique_labels,train_df,images,num_classes

def plot_img(X, y, index):
    plt.figure(figsize = (3,3))
    plt.imshow(X[index])
    plt.xlabel(unique_labels[int(y[index])])
    plt.show()

rows,cols = 1,5
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20,8))

for i in range(len(unique_labels)):
    img_path = train_df[train_df.categories == i].values[0]
    axes[i].set_title(img_path[0].split('/')[-2] + " - " + str(img_path[1]))
    axes[i].imshow(images[train_df[train_df.image_paths == img_path[0]].index[0]])

    plt.figure(figsize=(5, 5))

# Display the number of samples in each category
train_df.categories.value_counts().rename(index={
    0: unique_labels[0],
    1: unique_labels[1],
    2: unique_labels[2],
    3: unique_labels[3],
    4: unique_labels[4],
}).plot(kind='bar', color='darkturquoise')

plt.title('Number of Samples in Each Category')
plt.xticks(rotation=45)
plt.show()

# Display distribution of pixel values
plt.figure(figsize=(10, 6))
sns.histplot([img.mean() for img in images], bins=50, kde=True)
plt.title('Distribution of Mean Pixel Values')
plt.xlabel('Mean Pixel Value')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot([img[:, :, 0].mean() for img in images], bins=50, kde=True, color='red')
sns.histplot([img[:, :, 1].mean() for img in images], bins=50, kde=True, color='green')
sns.histplot([img[:, :, 2].mean() for img in images], bins=50, kde=True, color='blue')
plt.title('Distribution of Mean Pixel Values')
plt.xlabel('Mean Pixel Value')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(20, 12))
for a in range(5):
    n= images[train_df.categories.values == a]
    plt.subplot(2,3,a+1)
    sns.histplot([img[:, :, 0].mean() for img in n], bins=50, kde=True, color='red')
    sns.histplot([img[:, :, 1].mean() for img in n], bins=50, kde=True, color='green')
    sns.histplot([img[:, :, 2].mean() for img in n], bins=50, kde=True, color='blue')
    plt.title('Distribution of Mean Pixel Values of category {}'.format(unique_labels[a]))
    plt.xlabel('Mean Pixel Value')
    plt.ylabel('Frequency')
plt.show()

# Distribution of Mean Pixel Values by Category using displot
plt.figure(figsize=(15, 10))
for i in range(num_classes):
    mean_values = [img.mean() for img in images[train_df.categories.values == i]]
    sns.kdeplot(mean_values, label=unique_labels[i])
plt.title('Distribution of Mean Pixel Values by Category')
plt.xlabel('Mean Pixel Value')
plt.ylabel('Density')
plt.legend()
plt.show()

