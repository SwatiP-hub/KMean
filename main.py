from model import KMeans
from utils import get_image, show_image, save_image, error
import matplotlib.pyplot as plt

def main():
    # get image
    image = get_image('image.jpg')
    #print("Loaded image shape:", image.shape)
    img_shape = image.shape

    # reshape image
    image_flattened = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    #print("Flattened image shape:", image_flattened.shape) 
    # create model
    k_values = [2, 5, 10, 20, 50]
    mse_values = []

    # Iterate over different values of k
    for k in k_values:
        print(f"Running KMeans with k = {k}")
        kmeans = KMeans(k)
        kmeans.fit(image_flattened)

        # Replace each pixel with its closest cluster center
        clustered_image = kmeans.replace_with_cluster_centers(image_flattened)

        # Reshape the clustered image back to the original shape
        clustered_image_reshaped = clustered_image.reshape(img_shape)
       
        # Calculate the MSE
        mse = error(image, clustered_image_reshaped)
        mse_values.append(mse)
        print(f"MSE for k = {k}: {mse}")

        save_image(clustered_image_reshaped, f'image_clustered_{k}.jpg')
    
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, mse_values, marker='o', linestyle='-', color='b')
    plt.title("MSE vs Number of Clusters (k)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.grid(True)
    plt.show()
    plt.savefig('mse_vs_k_plot.png')  # Save the plot as an image
    print("Plot saved as mse_vs_k_plot.png")
  

    # show/save image
    #show_image(image)
    



if __name__ == '__main__':
    main()
