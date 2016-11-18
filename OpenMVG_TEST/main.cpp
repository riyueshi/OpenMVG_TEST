#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <io.h>
#include <math.h>
#include <iomanip>
#include <algorithm>

#include "openMVG/multiview/triangulation_nview.hpp"
#include "openMVG/sfm/sfm_data_triangulation.hpp"
#include "openMVG/geometry/rigid_transformation3D_srt.hpp"
#include "openMVG/geometry/Similarity3.hpp"
#include "openMVG/sfm/sfm_data_transform.hpp"
#include "openMVG/sfm/sfm.hpp"

#include <Eigen\Eigen>
#include <Eigen\Dense>
#include <Eigen\QR>

using namespace Eigen;

typedef Matrix<long double, 3, 4> Pmatrixld;
typedef Matrix<long double, 3, 3> Mat3x3ld;
typedef Matrix<long double, 3, 1> Vec3ld;

using namespace openMVG;
using namespace openMVG::cameras;
using namespace openMVG::geometry;
using namespace openMVG::image;
using namespace openMVG::sfm;

bool functionRQ(Pmatrixld pMat, Mat3x3ld &K, Mat3x3ld &R, Vec3ld &C)
{
	Mat3x3ld subPmat;
	subPmat << pMat(0, 0), pMat(0, 1), pMat(0, 2),
		pMat(1, 0), pMat(1, 1), pMat(1, 2),
		pMat(2, 0), pMat(2, 1), pMat(2, 2);
	//cout << "subPmat :" << endl << subPmat << endl;

	Mat3x3ld subPmatT = subPmat.transpose();
	Mat3x3ld subPmatUDLF;
	subPmatUDLF << subPmatT(2, 2), subPmatT(2, 1), subPmatT(2, 0),
		subPmatT(1, 2), subPmatT(1, 1), subPmatT(1, 0),
		subPmatT(0, 2), subPmatT(0, 1), subPmatT(0, 0);
	//FullPivHouseholderQR<Mat3x3ld> qr(3, 3);
	HouseholderQR<Mat3x3ld> qr(3, 3);
	//cout <<"subPmatUDLF :"<<endl<< subPmatUDLF << endl;
	qr.compute(subPmatUDLF);
	Mat3x3ld Rtemp = qr.matrixQR().triangularView<Upper>();
	//Mat3x3ld Qtemp = qr.matrixQ();
	Mat3x3ld Qtemp = qr.householderQ();
	Mat3x3ld QtempT = Qtemp.transpose();
	//Mat3x3ld Q;
	R << QtempT(2, 2), QtempT(2, 1), QtempT(2, 0),
		QtempT(1, 2), QtempT(1, 1), QtempT(1, 0),
		QtempT(0, 2), QtempT(0, 1), QtempT(0, 0);
	//Mat3x3ld R;
	Mat3x3ld RtempT = Rtemp.transpose();

	K << RtempT(2, 2), RtempT(2, 1), RtempT(2, 0),
		RtempT(1, 2), RtempT(1, 1), RtempT(1, 0),
		RtempT(0, 2), RtempT(0, 1), RtempT(0, 0);

	if (K(0, 0) < 0)
	{
		K(0, 0) = -K(0, 0);
		R(0, 0) = -R(0, 0);
		R(0, 1) = -R(0, 1);
		R(0, 2) = -R(0, 2);
	}
	if (K(1, 1) < 0)
	{
		K(0, 1) = -K(0, 1);
		K(1, 1) = -K(1, 1);
		R(1, 0) = -R(1, 0);
		R(1, 1) = -R(1, 1);
		R(1, 2) = -R(1, 2);
	}
	if (R.determinant() < 0)
	{
		R = -R;
	}
	Vec3ld lastColP;
	lastColP << pMat(0, 3), pMat(1, 3), pMat(2, 3);
	C = -(K*R).inverse()*lastColP;

	//cout << setiosflags(ios::fixed) << setprecision(13)
	//	<< "R : " << endl << R << endl << "K: " << endl << K << endl << "C: " << endl << C;
	return true;
}

bool get_filelist_from_dir(std::string path, vector<std::string>& files)
{
	long   hFile = 0;
	struct _finddata_t fileinfo;
	files.clear();
	if ((hFile = _findfirst(path.c_str(), &fileinfo)) != -1)
	{
		do
		{
			if (!(fileinfo.attrib &  _A_SUBDIR))
				files.push_back(fileinfo.name);
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
		return true;
	}
	else
		return false;
}

bool readInPmatFile(vector<std::string> &pMatNameVec, vector<Pmatrixld> &pMatVec)
{
	for (int i = 0; i < pMatNameVec.size(); i++)
	{
		ifstream pMatReader(pMatNameVec.at(i));
		if (!pMatReader)
		{
			std::cout << "open projection matrix " << pMatNameVec.at(i) << " failed!" << endl;
			getchar();
		}
		Pmatrixld pMat;
		std::string tempString;
		pMatReader >> tempString;
		pMatReader >> pMat(0, 0) >> pMat(0, 1) >> pMat(0, 2) >> pMat(0, 3)
			>> pMat(1, 0) >> pMat(1, 1) >> pMat(1, 2) >> pMat(1, 3)
			>> pMat(2, 0) >> pMat(2, 1) >> pMat(2, 2) >> pMat(2, 3);
		pMatVec.push_back(pMat);
	}

	return true;
}

int GetCorrespondRelation(const std::vector<string>&pMatNameVec,
	const SfM_Data& sfmData,
	std::vector<int> &correspondPMatIndex)
{
	int validCount(0);
	for (size_t viewIndex = 0; viewIndex < sfmData.views.size(); viewIndex++)
	{
		string imageName = sfmData.views.at(viewIndex)->s_Img_path;
		string correspondPMatName = imageName.substr(1, imageName.size() - 5) + ".txt";
		auto pos = find(pMatNameVec.begin(), pMatNameVec.end(), correspondPMatName);
		if (pos != pMatNameVec.end())
		{
			correspondPMatIndex.push_back(pos - pMatNameVec.begin());
			++validCount;
		}
		else
		{
			correspondPMatIndex.push_back(-1);
		}
	}
	return validCount;
}

bool GetSrcPointsAndTarPoints(const std::vector<string>&pMatNameVec,
	const std::vector<Pmatrixld> &pMatVec,
	const SfM_Data& sfmData,
	Mat &srcPoints,
	Mat &tarPoints)
{
	//if (pMatVec.size() != sfmData.GetPoses().size())
	//{
	//	std::cout << "Error: the number of camera center not equal!" << std::endl;
	//	return false;
	//}

	vector<int> pMatIndex;
	int numOfPoints = GetCorrespondRelation(pMatNameVec, sfmData, pMatIndex);
	for (size_t i = 0; i < pMatIndex.size(); i++)
	{
		std::cout << pMatIndex.at(i) << endl;
	}
	std::cout << "number of points :" << numOfPoints << endl;
	srcPoints = Mat(3, numOfPoints);
	tarPoints = Mat(3, numOfPoints);

	vector<double> xVec;
	vector<double> yVec;
	const double scaleParam(100.0);


	for (int i = 0, j = 0; i < pMatIndex.size(); i++)
	{
		if (pMatIndex.at(i)==-1)
		{
			continue;
		}
		Mat3x3ld R, K;
		Vec3ld C;
		functionRQ(pMatVec.at(pMatIndex.at(i)), K, R, C);
		//srcPoints(0, i) = C(0);
		//srcPoints(1, i) = C(1);
		//srcPoints(2, i) = C(2);

		//tarPoints(0, i) = (sfmData.GetPoses().at(i).center()(0) - transParams(0)) / scaleParam;
		//tarPoints(0, i) = (sfmData.GetPoses().at(i).center()(1) - transParams(1)) / scaleParam;
		//tarPoints(0, i) = (sfmData.GetPoses().at(i).center()(2);
		//tarPoints(0, i) = (C(0) - transParams(0)) / scaleParam;
		//tarPoints(1, i) = (C(1) - transParams(1)) / scaleParam;
		//tarPoints(2, i) = (C(2)) / scaleParam;
		xVec.push_back(C(0));
		yVec.push_back(C(1));
		tarPoints(0, j) = C(0);
		tarPoints(1, j) = C(1);
		tarPoints(2, j) = C(2);
		srcPoints(0, j) = sfmData.GetPoses().at(i).center()(0);
		srcPoints(1, j) = sfmData.GetPoses().at(i).center()(1);
		srcPoints(2, j) = sfmData.GetPoses().at(i).center()(2);
		j++;
	}


	double transX = std::accumulate(xVec.begin(), xVec.end(), 0.0) / xVec.size();
	double transY = std::accumulate(yVec.begin(), yVec.end(), 0.0) / yVec.size();
	for (size_t i = 0; i < pMatVec.size(); i++)
	{
		tarPoints(0, i) = (tarPoints(0, i) - transX) / scaleParam;
		tarPoints(1, i) = (tarPoints(1, i) - transY) / scaleParam;
		tarPoints(2, i) = tarPoints(2, i) / scaleParam;
	}
	return true;
}

bool ControlPointRegistrate(const Mat& srcPointsVec, const Mat& tarPointsVec, SfM_Data &sfmData)
{

	//std::cout << "Original points coords:\n"
	//	<< srcPointsVec << std::endl << std::endl
	//	<< "Control points coords:\n"
	//	<< tarPointsVec << std::endl << std::endl;

	Vec3 t;
	Mat3 R;
	double S;
	if (openMVG::geometry::FindRTS(srcPointsVec, tarPointsVec, &S, &t, &R))
	{
		openMVG::geometry::Refine_RTS(srcPointsVec, tarPointsVec, &S, &t, &R);
		std::cout << "Found transform:\n"
			<< " scale: " << S << "\n"
			<< " rotation:\n" << R << "\n"
			<< " translation: " << t.transpose() << std::endl;


		//--
		// Apply the found transformation as a 3D Similarity transformation matrix // S * R * X + t
		//--

		const openMVG::geometry::Similarity3 sim(geometry::Pose3(R, -R.transpose() * t / S), S);
		openMVG::sfm::ApplySimilarity(sim, sfmData);
	}
	else
		return false;

	//ofstream resWriter("res.txt");
	//for (size_t i = 0; i < sfmData.views.size(); i++)
	//{
	//	resWriter << tarPointsVec(0, i) - sfmData.poses.at(i).center()(0) << " "
	//		<< tarPointsVec(1, i) - sfmData.poses.at(i).center()(1) << " "
	//		<< tarPointsVec(2, i) - sfmData.poses.at(i).center()(2) << std::endl;
	//}
	//resWriter.close();
	return true;
}

bool TransferPointCloud()
{
	// Read the input SfM scene
	SfM_Data sfmData;
	string sSfMDataFilename = "sfm_data.bin";
	if (!Load(sfmData, sSfMDataFilename, ESfM_Data(ALL))) {
		std::cerr << std::endl
			<< "The input SfM_Data file \"" << sSfMDataFilename << "\" cannot be read." << std::endl;
		return EXIT_FAILURE;
	}
	std::cout << sfmData.GetPoses().size();

	//cout << endl << sfmData.GetViews().at(0)->s_Img_path << endl;
	//getchar();
	std::vector<std::string> pmatFileName;
	std::string pmatFilePath = "data47";
	std::string searchPath = "data47\\*.txt";
	get_filelist_from_dir(searchPath, pmatFileName);
	std::vector<std::string> pmatFileNameFull;
	pmatFileNameFull.resize(pmatFileName.size());

	for (int i = 0; i < pmatFileName.size(); i++)
	{
		pmatFileNameFull.at(i) = pmatFilePath + "\\" + pmatFileName.at(i);
	}
	std::vector<Pmatrixld> pMatVec;
	//cout << pmatFileName.size() << endl;
	//cout << pmatFileName.at(0) << endl;
	//getchar();
	readInPmatFile(pmatFileNameFull, pMatVec);
	Mat srcPoints, tarPoints;
	GetSrcPointsAndTarPoints(pmatFileName, pMatVec, sfmData, srcPoints, tarPoints);
	ControlPointRegistrate(srcPoints, tarPoints, sfmData);

	if (!Save(sfmData, "sfmDataGCP47.bin", ESfM_Data(ALL))) {
		std::cerr << std::endl
			<< "The output SfM_Data file \"" << "sfmDataGCP" << "\" cannot be write." << std::endl;
		return EXIT_FAILURE;
	}

	Save(sfmData,
		stlplus::create_filespec("data", "cloud_and_poses47", ".ply"),
		ESfM_Data(ALL));
	std::cout << "transfer finished, press to exit" << endl;
	getchar();
	return true;
}

namespace CamParams
{
	const string CamA("8176;6132;13681.91273;0;4065.975693;0;13681.91273;3096.7931703;0;0;1");
	const string CamB("8176;6132;13706.19335;0;4083.15818;0;13706.19335;3088.110210;0;0;1");
	const string CamC("8176;6132;13729.674848;0;4111.6916798;0;13729.6748483;3086.79781299;0;0;1");
	const string CamD("8176;6132;13676.202432;0;4094.636637;0;13676.202432;3074.08757945;0;0;1");
	const string CamE("8176;6132;8442.390494;0;4089.7757716;0;8442.39049452;3068.12360298;0;0;1");
}

bool GenerateImageListFile(const string imagePath)
{
	string searchPath = imagePath + "\\*.jpg";
	vector<string> imageNames;
	get_filelist_from_dir(searchPath, imageNames);

	ofstream listWriter("lists.txt");
	for (size_t imageIndex = 0; imageIndex < imageNames.size(); imageIndex++)
	{
		string camTag = imageNames.at(imageIndex).substr(3, 1);
		if (camTag == "A")
		{
			listWriter << imageNames.at(imageIndex) + ";" << CamParams::CamA << endl;
		}
		else if (camTag == "B")
		{
			listWriter << imageNames.at(imageIndex) + ";" << CamParams::CamB << endl;
		}
		else if (camTag == "C")
		{
			listWriter << imageNames.at(imageIndex) + ";" << CamParams::CamC << endl;
		}
		else if (camTag == "D")
		{
			listWriter << imageNames.at(imageIndex) + ";" << CamParams::CamD << endl;
		}
		else if (camTag == "E")
		{
			listWriter << imageNames.at(imageIndex) + ";" << CamParams::CamE << endl;
		}
		else
		{
			std::cout << "Error: can not write camera infomation for image " << imageNames.at(imageIndex)
				<< "Press Enter to exit" << endl;
			getchar();
			return false;
		}
	}
	return true;
}

int main()
{
	//GenerateImageListFile("D:\\riyueshi\\image303jpg");
	TransferPointCloud();
	return 0;
}