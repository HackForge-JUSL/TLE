import cv2 as cv
import torch
import mediapipe as mp
import torchvision
from torch import nn

labels = ['A',
 'B',
 'C',
 'D',
 'E',
 'F',
 'G',
 'H',
 'I',
 'J',
 'K',
 'L',
 'M',
 'N',
 'O',
 'P',
 'Q',
 'R',
 'S',
 'T',
 'U',
 'V',
 'W',
 'X',
 'Y',
 'Z',
 'space']

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

device = "cpu"


def load_model(model_path):
    with torch.no_grad():
        predictor = torchvision.models.googlenet(weights = False)
        predictor.aux1 = None
        predictor.aux2 = None #LeNet
        predictor.fc = nn.Sequential(
            nn.Linear(in_features=1024,out_features=256),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256,out_features=27)
        )
        state_dict = torch.load(model_path,map_location="cpu")
        predictor.load_state_dict(state_dict)
        predictor.to(device)
        predictor.eval()
        return predictor


def predict_letter(model,img):
    with torch.no_grad():
        img = torch.Tensor(img)
        img = img.reshape(3,224,224)
        img = img/255.0
        img = img.unsqueeze(dim=0)

        gs=model(img)
        tags=torch.softmax(gs,dim=1)
        tag=torch.argmax(tags,dim=1)
        return tag.item()


def find_hands(img):
    model = load_model("Sign Language Model_GoogleNet_250000.pth")
    img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    h,w,c = img.shape
    results = hands.process(img_rgb)
    comm = []
    letter = ""

    if results.multi_hand_landmarks:
        for handLMs in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLMs , mpHands.HAND_CONNECTIONS)
            x_max, y_max, x_min, y_min = 0, 0, w, h

            for lm in handLMs.landmark:
                x, y = int(lm.x*w), int(lm.y*h)

                if x > x_max:
                    x_max = x

                if x < x_min:
                    x_min = x

                if y > y_max:
                    y_max = y

                if y < y_min:
                    y_min = y

            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20

            cv.rectangle(img, (x_min, y_min) , (x_max,y_max) , (0,255,0) ,2)
            img2 = img[y_min:y_max, x_min:x_max]

            if img2.size != 0:
                img2 = cv.resize(img2, (224,224))
                letter = predict_letter(model,img2)
                letter = labels[letter]
                cv.putText(img, letter, (x_min + 5, y_max + 5), cv.FONT_HERSHEY_TRIPLEX, 3, (0, 255, 0),2)

                '''
                if letter == "space":
                    word = list(set(comm))[-1]
                    cv.putText(img,word,(x_min+5,y_max+5),cv.FONT_HERSHEY_TRIPLEX,3,(0,255,0),2 )
                    comm = []
                else:
                    comm.append(letter)'''

    return img


def main():
    cap = cv.VideoCapture(0)
    while True:
        success, img = cap.read()
        img = cv.flip(img,1)
        img = find_hands(img)

        cv.imshow("Image", img)
        key = cv.waitKey(1)
        if key & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()











