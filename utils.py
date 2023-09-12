import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


def char_to_pixels(text, path='arialbd.ttf', fontsize=20):
    """
    Based on https://stackoverflow.com/a/27753869/190597 (jsheperd)
    """
    font = ImageFont.truetype(path, fontsize) 
    w, h = font.getsize(text)  
    h *= 2
    image = Image.new('L', (w, h), 1)  
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font) 
    arr = np.asarray(image)
    arr = np.where(arr, 0, 1)
    arr = arr[(arr != 0).any(axis=1)]
    return arr

def Plot_loss_curve(train_list, test_dict):
    x_tst = list(test_dict.keys())
    y_tst = list(test_dict.values())
    train_x_vals = np.arange(len(train_list))
    plt.figure(2)
    plt.xlabel('Num Steps')
    plt.ylabel('ELBO')
    plt.title('ELBO Loss Curve')
    plt.plot(train_x_vals, train_list, label='train')
    plt.plot(x_tst, y_tst, label='tst')
    plt.legend(loc='best')
    plt.locator_params(axis='x', nbins=10)

    plt.show()
    return

def create_canvas(x, input_shape, batch_size, pred_rfs=[], comments=[]):
    
    number_of_mris = len(x) * batch_size
    
    plt.figure(1)
    input_len = np.max(input_shape)
    canvas = np.zeros((number_of_mris * input_len, 4 * input_len))
    for i in range(len(x)):
        for j in range(batch_size):
            canvas[(i * batch_size + j)*input_len : (i * batch_size + j)*input_len + input_shape[1], 0*input_len:0*input_len + input_shape[2]] = np.flipud(np.array(x[i][j, 75, :, :]))
            canvas[(i * batch_size + j)*input_len : (i * batch_size + j)*input_len + input_shape[0], 1*input_len:1*input_len + input_shape[2]] = np.flipud(np.array(x[i][j, :, 64, :]))
            canvas[(i * batch_size + j)*input_len : (i * batch_size + j)*input_len + input_shape[0], 2*input_len:2*input_len + input_shape[1]] = np.flipud(np.array(x[i][j, :, :, 53]))
            canvas[(i * batch_size + j)*input_len : (i * batch_size + j)*input_len + input_shape[0], 3*input_len:3*input_len + input_shape[1]] = np.flipud(np.array(x[i][j, :, :, 75]))
            
            if len(comments) > 0:
                rfs = ""
                rfs += comments[i][j]
                rfs = char_to_pixels(rfs)
                canvas[(i * batch_size + j)*input_len + input_len - rfs.shape[0] : (i * batch_size + j)*input_len + input_len,\
                    4*input_len - rfs.shape[1] : 4*input_len]\
                    = rfs

            if len(pred_rfs) > 0 and len(pred_rfs[i]) > 0:
                rfs = ""
                for rf in pred_rfs[i]:
                    pred = pred_rfs[i][rf][j].detach().cpu().numpy()
                    if len(pred) == 1:
                        rfs += "{:.2f}  ".format(pred.item())
                    else:
                        rfs += "{}  ".format(np.argmax(pred))
                rfs = char_to_pixels(rfs)
                canvas[(i * batch_size + j)*input_len + input_len - rfs.shape[0] : (i * batch_size + j)*input_len + input_len,\
                    4*input_len - rfs.shape[1] : 4*input_len]\
                    = rfs

    return canvas
