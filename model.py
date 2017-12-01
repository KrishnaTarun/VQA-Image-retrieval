#python model selection file

def get_model(model, nwords):
    img_list = [0]*10

    if model == "simple":
        from models.Deep_CBOW_simple import DeepCBOW
        model = DeepCBOW(nwords, img_list, 300, 2048, 64, 1)

    if model == "vectorized":
        from models.Deep_CBOW_vectorized import DeepCBOW
        model = DeepCBOW(nwords, img_list, 300, 2048, 64, 1)

    if model == "minibatch":
        from models.Deep_CBOW_minibatch import DeepCBOW
        model = DeepCBOW(nwords, 300, 2048, 64, 1)

    return model
