
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <set>

#include "camFusion.hpp"
#include "dataStructures.h"

#include "kdtree.h"


using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


void clusterHelper(int indice, std::vector<std::vector<float>> points, std::vector<int>& cluster, std::vector<bool>& processed, std::shared_ptr<KdTree> tree, float distanceTol)
{

    // Implement max and min size checking in Search function
	processed[indice] = true;
	cluster.push_back(indice);

	std::vector<int> nearest = tree->search(points[indice], distanceTol);

	for(int id : nearest)
	{
		if(!processed[id])
			clusterHelper(id, points, cluster, processed, tree, distanceTol);
	}
}


std::vector<std::vector<int>> euclideanCluster(std::vector<std::vector<float>> points, std::shared_ptr<KdTree> tree, float distanceTol, int minSize)
{
    std::vector<std::vector<int>> clusters;
    std::vector<bool> processed(points.size(), false);

    int i = 0;

    while(i < points.size())
    {
        if(processed[i])
        {
            i++;
            continue;
        }

        std::vector<int> cluster;

        clusterHelper(i, points, cluster, processed, tree, distanceTol);
        if(cluster.size() >= minSize)
            clusters.push_back(cluster);
        i++;
    }

    return clusters;
}


std::vector<LidarPoint> FilterOutliers(std::vector<LidarPoint> &lidarPoints, float clusterTolerance, int minSize)
{


    std::shared_ptr<KdTree> tree (std::make_shared<KdTree>());

    std::vector<LidarPoint> filteredPoints;

    std::vector<std::vector<float>> lidarPointsXYZ(lidarPoints.size());

    for (int i = 0; i<lidarPoints.size(); i++)
    {
        lidarPointsXYZ[i].push_back(static_cast<float>(lidarPoints[i].x));
        lidarPointsXYZ[i].push_back(static_cast<float>(lidarPoints[i].y));
        lidarPointsXYZ[i].push_back(static_cast<float>(lidarPoints[i].z));
    }

    tree->insertOptimized(lidarPointsXYZ);

    std::vector<std::vector<int>> clusters = euclideanCluster(lidarPointsXYZ, tree, clusterTolerance, minSize);



    std::cout << "Number of detected clusters: " << clusters.size()<< std::endl;

    for(const auto& cluster : clusters)
    {

        std::cout << "Number of detected points per cluster: " << cluster.size()<< std::endl;

        // std::vector<LidarPoint> clusteredPoints;
        for (const auto& point : cluster)
        {
            filteredPoints.push_back(lidarPoints[point]);
        }
        // if(clusteredPoints.size() > filteredPoints.size())
        //     filteredPoints = std::move(clusteredPoints);
    
    }

    return filteredPoints;

}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, 
                            std::vector<cv::KeyPoint> &kptsPrev, 
                            std::vector<cv::KeyPoint> &kptsCurr,        
                            std::vector<cv::DMatch> &kptMatches)
{

    double dist, sum, mean_distance = 0;
    vector<pair<cv::DMatch, double>> distance;

    for(auto it = kptMatches.begin(); it < kptMatches.end(); it++)
    {
        if(boundingBox.roi.contains(kptsCurr[it->trainIdx].pt))
        {
            pair<cv::DMatch, double> pair;
            dist = cv::norm(kptsCurr[it->trainIdx].pt - kptsPrev[it->queryIdx].pt);
            pair.first = (*it);
            pair.second = dist;
            distance.push_back(pair);
            sum += dist;         
        }            
    }
    mean_distance = sum / distance.size(); 

    for (auto it = distance.begin(); it != distance.end(); it++)
    {   
        if((*it).second < mean_distance * 1.3)
            boundingBox.kptMatches.push_back((*it).first);
    }
}
     
void clusterKptMatchesWithROI_2(BoundingBox &boundingBox, 
                            std::vector<cv::KeyPoint> &kptsPrev, 
                            std::vector<cv::KeyPoint> &kptsCurr,        
                            std::vector<cv::DMatch> &kptMatches)
{
    // 1. Find all keypoint matches that belong to each 3D object. This can be done by checkin if corresponding keypoints are within RegionOfInterest in
    // the camera image. All matches that are within the ROI can be added to a vector.

    // 2. There will be outliers in the matches and these need to be eliminated. To do this we can compute the robust mean od all Euclidean distances between
    // keypoint matches and remove those that are too far away from the mean.

    // 3. When all the keypoint matches are within the bounding boxes we can compute the TTC based on Camera image.

// Calculating an average distance between all matches and buffering all keypoint matches that are within the Bounding
// boxes Region-Of-Interest

    double dist, distance, mean_distance, match_distance = 0;
    std::vector<cv::DMatch> kptMatchesBuffer;

    for(auto it = kptMatches.begin(); it < kptMatches.end(); it++)
    {
        if(boundingBox.roi.contains(kptsCurr[it->trainIdx].pt))
        {
            kptMatchesBuffer.push_back(*it);
            dist += cv::norm(kptsCurr[it->trainIdx].pt - kptsPrev[it->queryIdx].pt);
        }            
    }
    mean_distance = dist / kptMatchesBuffer.size(); 

    // Pushing back all the keypoint matches whose distance is not bigger than Mean distance x multiplier
    for(auto it = kptMatchesBuffer.begin(); it < kptMatchesBuffer.end(); it++)
    {
        match_distance = cv::norm(kptsCurr[it->trainIdx].pt - kptsPrev[it->queryIdx].pt);

        if(match_distance < mean_distance * 1.3 )
        {
             boundingBox.kptMatches.push_back(*it);
        }
    }
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC_v0, cv::Mat *visImg)
{ 
    double dT = 1/frameRate;
    double medianDistRatio;
    vector<double> distRatios;

    for(auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; it1++)
    {   // outer keypoint loop

        // get current keypoint and its matched partner in the previous frame

        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); it2++)
        {   // inner keypoint look

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev frame

            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distancee ratios

            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            {   // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC_v0 = NAN;
        return;
    }

    sort(distRatios.begin(), distRatios.end());

    if (distRatios.size() % 2 == 0)
        medianDistRatio = (distRatios[distRatios.size() / 2] + distRatios[(distRatios.size() / 2) + 1]) / 2;
    else
        medianDistRatio = distRatios[(distRatios.size() + 1) / 2];

    TTC_v0 = - dT / (1 - medianDistRatio);
    std::cout << "Camera TTC = " << TTC_v0 << std::endl;

}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC_v0)
{

    double dT = 1/frameRate;
    double laneWidth = 4;
    float clusterTolerance = 0.04;
    int minSize = 50; 
    int minXCurr_count = 0;
    int minXPrev_count = 0;

    double minXPrev = 1e9, minXCurr = 1e9;

    std::vector<LidarPoint> lidarPointsPrevFiltered = FilterOutliers(lidarPointsPrev, clusterTolerance, minSize);
    std::vector<LidarPoint> lidarPointsCurrFiltered = FilterOutliers(lidarPointsCurr, clusterTolerance, minSize);


    for (auto it = lidarPointsPrevFiltered.begin(); it != lidarPointsPrevFiltered.end(); ++it)
        minXPrev = ((minXPrev > it->x) && (abs(it->y) <= laneWidth/2))? it->x : minXPrev;




    for (auto it = lidarPointsCurrFiltered.begin(); it != lidarPointsCurrFiltered.end(); ++it)
       minXCurr = ((minXCurr > it->x) && (abs(it->y) <= laneWidth/2)) ? it->x : minXCurr;


    std::cout << "Previos distance to the car: " << minXPrev << std::endl;
    std::cout << "Currrent distance to the car: " << minXCurr << std::endl;

    TTC_v0 = minXCurr * dT / (minXPrev - minXCurr);
    std::cout << "LiDAR TTC = " << TTC_v0 << std::endl;

    // TODO 
    // Constant Acceleration Model
    // TTC_a0 = ; 
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
   
    std::multimap<int, int> bounding_boxes_matches;
    std::set<int> prev_frame_box_in_kpt;    

    int values_num = 0;

    for (auto it= matches.begin(); it < matches.end(); it++)
    {
        // for every match there are 2 keypoints. Find by all bounding boxes enclosing the keypoints in previous and current image
        // and put them in the multimap
        cv::KeyPoint keypoint_match_prev_frame = prevFrame.keypoints[it->queryIdx]; // keypoint matched on the previous frame
        cv::KeyPoint keypoint_match_curr_frame = currFrame.keypoints[it->trainIdx]; // keypoint matched on the current frame
       
        for (auto it_p = prevFrame.boundingBoxes.begin(); it_p < prevFrame.boundingBoxes.end(); it_p++)
        {
            if(it_p -> roi.contains(keypoint_match_prev_frame.pt))
            {
                for (auto it_c = currFrame.boundingBoxes.begin(); it_c < currFrame.boundingBoxes.end(); it_c++)
                {
                    if(it_c -> roi.contains(keypoint_match_curr_frame.pt))
                    {                        
                        prev_frame_box_in_kpt.insert(it_p->boxID);
                        bounding_boxes_matches.insert({it_p->boxID, it_c->boxID});
                        values_num = std::max(it_c->boxID, values_num);
                    }             
                }
            }             
        }
    }
  
    for(auto it = prev_frame_box_in_kpt.begin(); it != prev_frame_box_in_kpt.end(); it++ )
    {
        std::vector<int> maxID (values_num + 1, 0);

        for(auto it_mm = bounding_boxes_matches.equal_range(*it).first; it_mm != bounding_boxes_matches.equal_range(*it).second; ++it_mm )
        {
            maxID[(*it_mm).second]++;  
        }

        int mode = distance(maxID.begin(), max_element(maxID.begin(), maxID.end()));

        bbBestMatches.insert({*it, mode});      
    }
 
}
