#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv )
{
    if ( argc < 3 )
    {
        printf("usage: SGBM <images> <calib> <start_frame>\n");
        return -1;
    }
    int start_frame = 0;
    if (argc == 4)
        start_frame = atoi(argv[3]);

    Mat cam1,cam2;
    Mat dis1,dis2;
    Mat rot,trans;
    Mat E,F; 
    Mat R1,R2;
    Mat P1,P2;
    Mat Q;
    Size imageSize;
    FileStorage fs(argv[2], FileStorage::READ );
    fs["cam1"] >> cam1;
    fs["dis1"] >>  dis1;
    fs["cam2"] >>  cam2;
    fs["dis2"] >>  dis2;
    fs["imageSize"] >>  imageSize;
    fs["rot"] >>  rot;
    fs["trans"] >>  trans;
    fs["R1"] >>  R1;
    fs["R2"] >>  R2;
    fs["P1"] >>  P1;
    fs["P2"] >>  P2;
    fs["Q"] >>  Q;

    fs.release();
    cam1 = cam1*0.5;
    cam2 = cam2*0.5;

    int offset_X = 160;
    int offset_Y = 14;

    cam1.at<double>(2,2) = cam2.at<double>(2,2) = 1;

    imageSize = Size(imageSize.width*0.5,imageSize.height*0.5);

    double alpha = 1;
    stereoRectify(cam1,dis1,cam2,dis2,imageSize,rot,trans,R1,R2,P1,P2,Q,CALIB_ZERO_DISPARITY,alpha,imageSize,0,0);

    VideoCapture sequenceLeft(argv[1]);
    sequenceLeft.set(CV_CAP_PROP_POS_FRAMES, start_frame);

    Mat rmap[2][2];
    initUndistortRectifyMap(cam1, dis1, R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cam2, dis2, R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    int minDisparity = 10;//minimaler Disparitäten Abstand
    int maxDisparity = 160;//maximaler Disparitäten Abstand (mehrfaches von 16)
    int blocksize = 3;
    //Parameter für Glattheit P1 < P2
    int p1 = 8*3*blocksize*blocksize;
    int p2 = 32*3*blocksize*blocksize;


    Ptr<StereoSGBM> sgbm = StereoSGBM::create(minDisparity, maxDisparity, blocksize);
    // setting the penalties for SGBM
    sgbm->setP1(p1);
    sgbm->setP2(p2);
    sgbm->setMinDisparity(minDisparity);
//     sgbm->setUniquenessRatio(5);
//     sgbm->setSpeckleWindowSize(400);
//     sgbm->setSpeckleRange(0);
//     sgbm->setDisp12MaxDiff(1);
//     sgbm->setBinaryKernelType(binary_descriptor_type);
//     sgbm->setSpekleRemovalTechnique(CV_SPECKLE_REMOVAL_AVG_ALGORITHM);
//     sgbm->setSubPixelInterpolationMethod(CV_SIMETRICV_INTERPOLATION);
//     Alternative for scalling
//     imgDisparity16S2.convertTo(imgDisparity8U2, CV_8UC1, scale);

    Mat imgLeft, imgRight, imgLeftRec, imgRightRec, disp;
    Mat mask = Mat::ones(imageSize,CV_8U)*255, mask_rec;
    remap(mask,mask_rec, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);

    Mat image;
    for (;;)
    {
        sequenceLeft >> image;
        resize(image, image, Size(image.cols, image.rows/2), INTER_NEAREST);

        Rect box(offset_X, offset_Y, 640,512);
        imgRight = image(box);
        box.x+=image.cols/2;
        imgLeft = image(box);

        if (imgLeft.empty()||imgRight.empty())
            break;

        remap(imgLeft, imgLeftRec, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);
        remap(imgRight,imgRightRec,rmap[1][0], rmap[1][1], CV_INTER_LINEAR);

        Mat disp_mask;

        sgbm -> compute(imgLeftRec,imgRightRec, disp);

        disp.copyTo(disp_mask, mask_rec);
        double min, max;
        cv::minMaxLoc(disp_mask, &min, &max);
        float alpha = 255.f/(max - min);
        float beta = -min*alpha;
        Mat disp2;
        disp_mask.convertTo(disp2,CV_8U,alpha,beta);

        Mat imgLeftRecMask;
        imgLeftRec.copyTo(imgLeftRecMask, mask_rec);

        imshow("Left", imgLeft);
        imshow("Right", imgRight);
        imshow("LeftRec", imgLeftRec);
        imshow("RightRec", imgRightRec);
        imshow("Disp", disp2);
        waitKey(1);
    }
    return 0;
}
