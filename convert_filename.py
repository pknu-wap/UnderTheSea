import glob
import cv2

def main():
    file_path = input('사진이 있는 폴더 경로 : ')
    converted_path = input('이름 변경 후 저장할 폴더 경로 : ')

    path = sorted(glob.glob(file_path+'/*'))


    for i, file_name in enumerate(path):
        try :
            img = cv2.imread(file_name, cv2.IMREAD_COLOR)
            if img.shape[2] == 3:
                print(file_name + ' -> ' + converted_path+'/참돔%04d.jpg'% (i+1))
                cv2.imwrite(converted_path+ '/'+ '참돔%04d.jpg'% (i+1), img)
            else:
                print('RGB 파일이 아닌 것 : ', file_name)
                
        except AttributeError as e:
            print(e, file_name)
    
if __name__ == "__main__":
    main()
