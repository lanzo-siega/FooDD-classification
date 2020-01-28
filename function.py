#image path
check = '/path/to/image'

#classification function
def categorize(file):

  # resizing and greyscaling the image so that the model can read the data
  def prepare(file):
    img_size = 50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 1)
  
  #calling the model that was trained earlier
  model = tf.keras.models.load_model("CNN.model")
  image = prepare(check) 
  prediction = model.predict([image])
  prediction = list(prediction[0])
  print(cat[prediction.index(max(prediction))])
