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

	if (K(0, 0)<0)
	{
		K(0, 0) = -K(0, 0);
		R(0, 0) = -R(0, 0);
		R(0, 1) = -R(0, 1);
		R(0, 2) = -R(0, 2);
	}
	if (K(1, 1)<0)
	{
		K(0, 1) = -K(0, 1);
		K(1, 1) = -K(1, 1);
		R(1, 0) = -R(1, 0);
		R(1, 1) = -R(1, 1);
		R(1, 2) = -R(1, 2);
	}
	if (R.determinant()<0)
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
			cout << "open projection matrix " << pMatNameVec.at(i) << " failed!" << endl;
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

bool GetSrcPointsAndTarPoints(const std::vector<Pmatrixld> &pMatVec, const SfM_Data& sfmData, Mat &srcPoints, Mat &tarPoints)
{
	if (pMatVec.size()!=sfmData.GetPoses().size())
	{
		std::cout << "Error: the number of camera center not equal!" << std::endl;
		return false;
	}
	const int numOfPoints(pMatVec.size());
	srcPoints = Mat(3, numOfPoints);
	tarPoints = Mat(3, numOfPoints);

	Mat3x3ld R, K;
	Vec3ld C;
	functionRQ(pMatVec.at(11), K, R, C);
	//Vec3 transParams = Vec3(C(0),C(1),C(2));
	vector<double> xVec;
	vector<double> yVec;
	const double scaleParam(100.0);

	for (int i = 0; i < pMatVec.size(); i++)
	{
		Mat3x3ld R, K;
		Vec3ld C;
		functionRQ(pMatVec.at(i), K, R, C);
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
		tarPoints(0, i) = C(0);
		tarPoints(1, i) = C(1);
		tarPoints(2, i) = C(2);

		srcPoints(0, i) = sfmData.GetPoses().at(i).center()(0);
		srcPoints(1, i) = sfmData.GetPoses().at(i).center()(1);
		srcPoints(2, i) = sfmData.GetPoses().at(i).center()(2);

	}


	double transX = std::accumulate(xVec.begin(), xVec.end(), 0.0) / xVec.size();
	double transY = std::accumulate(yVec.begin(), yVec.end(), 0.0) / yVec.size();
	for (size_t i = 0; i < pMatVec.size(); i++)
	{
		tarPoints(0, i) = (tarPoints(0, i) - transX)/ scaleParam;
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
	return true;
}



int main()
{
	// Read the input SfM scene
	SfM_Data sfmData;
	string sSfMDataFilename = "sfm_data.bin";
	if (!Load(sfmData, sSfMDataFilename, ESfM_Data(ALL))) {
		std::cerr << std::endl
			<< "The input SfM_Data file \"" << sSfMDataFilename << "\" cannot be read." << std::endl;
		return EXIT_FAILURE;
	}
	cout << sfmData.GetPoses().size();

	std::vector<std::string> pmatFileName;
	std::string pmatFilePath = "data47";
	std::string searchPath = "data47\\*.txt";
	get_filelist_from_dir(searchPath, pmatFileName);

	for (int i = 0; i < pmatFileName.size(); i++)
	{
		pmatFileName.at(i) = pmatFilePath + "\\" + pmatFileName.at(i);
	}
	std::vector<Pmatrixld> pMatVec;
	//cout << pmatFileName.size() << endl;
	//cout << pmatFileName.at(0) << endl;
	//getchar();
	readInPmatFile(pmatFileName, pMatVec);
	Mat srcPoints, tarPoints;
	GetSrcPointsAndTarPoints(pMatVec, sfmData, srcPoints, tarPoints);
	ControlPointRegistrate(srcPoints, tarPoints, sfmData);

	if (!Save(sfmData, "sfmDataGCP47.bin", ESfM_Data(ALL))) {
		std::cerr << std::endl
			<< "The output SfM_Data file \"" << "sfmDataGCP" << "\" cannot be write." << std::endl;
		return EXIT_FAILURE;
	}

	Save(sfmData,
		stlplus::create_filespec("data", "cloud_and_poses47", ".ply"),
		ESfM_Data(ALL));
	getchar();
	return 0;
}