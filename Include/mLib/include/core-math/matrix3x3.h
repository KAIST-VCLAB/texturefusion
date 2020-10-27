
#ifndef CORE_MATH_MATRIX3X3_H_
#define CORE_MATH_MATRIX3X3_H_

namespace ml {

template <class FloatType>
class DenseMatrix;
template <class FloatType>
struct EigenSystem;

//! This class provides functions to handle 3-dimensional matrices
/*! The arrangement of the matrix is row-like.
    The index of a specific position is:
    <pre>
       0  1  2
       3  4  5
	   6  7  8
    </pre>
*/
template <class FloatType> class Matrix3x3 : public BinaryDataSerialize< Matrix3x3<FloatType> >
{
public:
	//! An uninitialized matrix
	Matrix3x3() {}

	//! Initialize with values stored in an array
	Matrix3x3(const FloatType* values) {
		for (unsigned int i = 0; i < 9; i++) {
			matrix[i] = values[i];
		}
	}

	//! Initialize from 3 row vectors
	Matrix3x3(const point3d<FloatType>& v0, const point3d<FloatType>& v1, const point3d<FloatType>& v2) {
		matrix[0] = v0.x;	matrix[1] = v0.y;	matrix[2] = v0.z;
		matrix[3] = v1.x;	matrix[4] = v1.y;	matrix[5] = v1.z;
		matrix[6] = v2.x;	matrix[7] = v2.y;	matrix[8] = v2.z;
	}

	//! Initializes the matrix row wise
	Matrix3x3(	const FloatType &m00, const FloatType &m01, const FloatType &m02,
				const FloatType &m10, const FloatType &m11, const FloatType &m12,
				const FloatType &m20, const FloatType &m21, const FloatType &m22) 
	{
		_m00 = m00;	_m01 = m01;	_m02 = m02;
		_m10 = m10;	_m11 = m11;	_m12 = m12;
		_m20 = m20;	_m21 = m21;	_m22 = m22;
	}

	//! Initialize with a matrix from another type
	template<class U>
	Matrix3x3(const Matrix3x3<U>& other) {
		for (unsigned int i = 0; i < 9; i++) {
			matrix[i] = (FloatType)other.getRawData()[i];
		}
	}

	//! Access element of Matrix at row x and column y for constant access
	inline FloatType at(unsigned char x, unsigned char y) const {
		assert((x<3)&&(y<3)); // no test if x<0 or y<0; they are unsigned char
		return matrix2[x][y]; 
	}
	//! Access element of Matrix at row x and column y
	inline  FloatType& at(unsigned char x, unsigned char y) {
		assert((x<3)&&(y<3)); // no test if x<0 or y<0; they are unsigned char
		return matrix2[x][y]; 
	}

	//! Access i,j-th row of Matrix for constant access
	inline  FloatType operator() (unsigned int i, unsigned int j) const {
		assert(i < 3 && j < 3);
		return matrix2[i][j];
	}
	//! Access i,j-th element of Matrix
	inline  FloatType& operator() (unsigned int i, unsigned int j) {
		assert(i < 3 && j < 3);
		return matrix2[i][j]; 
	}

	//! Access i-th element of the Matrix for constant access
	inline  FloatType operator[] (unsigned int i) const {
		assert(i < 9);
		return matrix[i];
	}
	//! Access i-th element of the Matrix
	inline  FloatType& operator[] (unsigned int i) {
		assert(i < 9);
		return matrix[i];
	}


	//! overwrite the matrix with an identity-matrix
	void setIdentity() {
		setScale(1.0, 1.0, 1.0);
	}
	static Matrix3x3 identity() {
		Matrix3x3 res;	res.setIdentity();
		return res;
	}

	//! sets the matrix zero (or a specified value)
	void setZero(FloatType v = (FloatType)0) {
		matrix[0] = matrix[1] = matrix[2] = v;
		matrix[3] = matrix[4] = matrix[5] = v;
		matrix[6] = matrix[7] = matrix[8] = v;
	}
	static Matrix3x3 zero(FloatType v = (FloatType)0) {
		Matrix3x3 res; res.setZero(v);
		return res;
	}

	//! overwrite the matrix with a rotation-matrix around a coordinate-axis (angle is specified in degrees, ccw)
	void setRotationX(FloatType angle) {
		FloatType anglerad = math::degreesToRadians(angle);
		FloatType sinangle = sin(anglerad);
		FloatType cosangle = cos(anglerad);

		matrix[0]=1;  matrix[1]=0;         matrix[2]=0;         
		matrix[3]=0;  matrix[4]=cosangle;  matrix[5]=-sinangle; 
		matrix[6]=0;  matrix[7]=sinangle;  matrix[8]= cosangle;   
	}
	static Matrix3x3 rotationX(FloatType angle) {
		Matrix3x3 res;	res.setRotationX(angle);
		return res;
	}

	//! overwrite the matrix with a rotation-matrix around a coordinate-axis (angle is specified in degrees, ccw)
	void setRotationY(FloatType angle) {
		FloatType anglerad = math::degreesToRadians(angle);
		FloatType sinangle = sin(anglerad);
		FloatType cosangle = cos(anglerad);

		matrix[0]= cosangle;  matrix[1]=0;  matrix[2]=sinangle;
		matrix[3]=0;          matrix[4]=1;  matrix[5]=0;        
		matrix[6]=-sinangle;  matrix[7]=0;  matrix[8]=cosangle;     
	}
	static Matrix3x3 rotationY(FloatType angle) {
		Matrix3x3 res;	res.setRotationY(angle);
		return res;
	}

	//! overwrite the matrix with a rotation-matrix around a coordinate-axis (angle is specified in degrees, ccw)
	void setRotationZ(FloatType angle) {
		FloatType anglerad = math::degreesToRadians(angle);
		FloatType sinangle = sin(anglerad);
		FloatType cosangle = cos(anglerad);

		matrix[0]=cosangle;  matrix[1]=-sinangle;  matrix[2]=0; 
		matrix[3]=sinangle;  matrix[4]= cosangle;  matrix[5]=0;
		matrix[6]=0;         matrix[7]=0;          matrix[8]=1;  
	}
	static Matrix3x3 rotationZ(FloatType angle) {
		Matrix3x3 res;	res.setRotationZ(angle);
		return res;
	}

	//! overwrite the matrix with a rotation-matrix around a coordinate-axis (angle in degrees, ccw)
	void setRotation(FloatType yaw, FloatType pitch, FloatType roll) {
		*this = rotationY(yaw) * rotationX(pitch) * rotationZ(roll);
	}
	static Matrix3x3 rotation(FloatType yaw, FloatType pitch, FloatType roll) {
		Matrix3x3 res;	res.setRotation(yaw, pitch, roll);
		return res;
	}

	//! overwrite the matrix with a rotation-matrix around a coordinate-axis (angle is specified in degrees, ccw)
	void setRotation(const point3d<FloatType> &axis, FloatType angle) {
		FloatType anglerad = math::degreesToRadians(angle);
		FloatType c = cos(-anglerad);
		FloatType s = sin(-anglerad);
		FloatType t = (FloatType)1.0f - c;

		point3d<FloatType> normalizedAxis = axis.getNormalized();
		FloatType x = normalizedAxis.x;
		FloatType y = normalizedAxis.y;
		FloatType z = normalizedAxis.z;

		matrix2[0][0] = 1 + t*(x*x-1);
		matrix2[0][1] = z*s+t*x*y;
		matrix2[0][2] = -y*s+t*x*z;

		matrix2[1][0] = -z*s+t*x*y;
		matrix2[1][1] = 1+t*(y*y-1);
		matrix2[1][2] = x*s+t*y*z;

		matrix2[2][0] = y*s+t*x*z;
		matrix2[2][1] = -x*s+t*y*z;
		matrix2[2][2] = 1+t*(z*z-1);
	}

	//! generates a rotation matrix (angle in degree, ccw)
	static Matrix3x3 rotation(const point3d<FloatType> &axis, FloatType angle) {
		Matrix3x3 res;	res.setRotation(axis, angle);
		return res;
	}

	//! overwrite the matrix with a scale-matrix
	void setScale(FloatType x, FloatType y, FloatType z) {
		matrix[0] =   x; matrix[1] = 0.0; matrix[2]  = 0.0; 
		matrix[3] = 0.0; matrix[4] =   y; matrix[5]  = 0.0;
		matrix[6] = 0.0; matrix[7] = 0.0; matrix[8] =   z;
	}
	static Matrix3x3 scale(FloatType x, FloatType y, FloatType z) {
		Matrix3x3 res;	res.setScale(x,y,z);
		return res;
	}

	//! overwrite the matrix with a scale-matrix
	void setScale(FloatType s) {
		setScale(s,s,s);
	}
	static Matrix3x3 scale(FloatType s) {
		Matrix3x3 res;	res.setScale(s);
		return res;
	}
	
	//! overwrite the matrix with a scale-matrix
	void setScale(const point3d<FloatType>& v) {
		matrix[0] = v.x; matrix[1] = 0.0; matrix[2] = 0.0;
		matrix[3] = 0.0; matrix[4] = v.y; matrix[5] = 0.0;
		matrix[6] = 0.0; matrix[7] = 0.0; matrix[8] = v.z;
	}
	static Matrix3x3 scale(const point3d<FloatType>& v) {
		Matrix3x3 res;	res.setScale(v);
		return res;
	}
	
	
	//! overwrite the matrix with a diagonal matrix
	void setDiag(FloatType x, FloatType y, FloatType z) {
		setScale(x,y,z);
	}
	static Matrix3x3 diag(FloatType x, FloatType y, FloatType z) {
		Matrix3x3 res;	res.setDiag(x,y,z);
		return res;
	}
	FloatType trace() const {
		return (matrix2[0][0] + matrix2[1][1] + matrix2[2][2]);
	}



	//! return the product of the operand with matrix
	Matrix3x3 operator* (const Matrix3x3& other) const {
		Matrix3x3<FloatType> result;
		//TODO unroll the loop
		for (unsigned char i = 0; i < 3; i++) {
			for (unsigned char j = 0; j < 3; j++) {
				result.at(i,j) = 
					this->at(i,0) * other.at(0,j) + 
					this->at(i,1) * other.at(1,j) + 
					this->at(i,2) * other.at(2,j);
			}
		}
		return result;
	}
	//! multiply operand with matrix b
	Matrix3x3& operator*= (const Matrix3x3& other) {
		Matrix3x3<FloatType> prod = (*this)*other;
		*this = prod;
		return *this;
	}
	//! multiply each element in the matrix with a scalar factor
	Matrix3x3 operator* (FloatType r) const {
		Matrix3x3<FloatType> result;
		for (unsigned int i = 0; i < 9; i++) {
			result.matrix[i] = matrix[i] * r;
		}
		return result;
	}
	//! multiply each element in the matrix with a scalar factor
	Matrix3x3& operator*= (FloatType r) {
		for (unsigned int i = 0; i < 9; i++) {
			matrix[i] *= r;
		}
		return *this;
	}
	//! divide the matrix by a scalar factor
	Matrix3x3 operator/ (FloatType r) const {
		Matrix3x3<FloatType> result;
		for (unsigned int i = 0; i < 9; i++) {
			result.matrix[i] = matrix[i] / r;
		}
		return result;
	}
	//! divide each element in the matrix with a scalar factor
	Matrix3x3& operator/= (FloatType r) {
		for (unsigned int i = 0; i < 9; i++) {
			matrix[i] /= r;
		}
		return *this;
	}
	//! transform a 3D-vector with the matrix
	point3d<FloatType> operator* (const point3d<FloatType>& v) const {
		return point3d<FloatType>(
			matrix[0]*v[0] + matrix[1]*v[1] + matrix[2]*v[2],
			matrix[3]*v[0] + matrix[4]*v[1] + matrix[5]*v[2],
			matrix[6]*v[0] + matrix[7]*v[1] + matrix[8]*v[2]
		);
	}
	//! return the sum of the operand with matrix b
	Matrix3x3 operator+ (const Matrix3x3& other) const {
		Matrix3x3<FloatType> result;
		for (unsigned int i = 0; i < 9; i++) {
			result.matrix[i] = matrix[i] + other.matrix[i];
		}
		return result;
	}

	//! add matrix other to the operand
	Matrix3x3& operator+= (const Matrix3x3& other) {
		for (unsigned int i = 0; i < 9; i++) {
			matrix[i] += other.matrix[i];
		}
		return *this;
	}

	//! return the difference of the operand with matrix b
	Matrix3x3 operator- (const Matrix3x3& other) const {
		Matrix3x3<FloatType> result;
		for (unsigned int i = 0; i < 9; i++) {
			result.matrix[i] = matrix[i] - other.matrix[i];
		}
		return result;
	}
	//! subtract matrix other from the operand
	Matrix3x3 operator-= (const Matrix3x3& other) {
		for (unsigned int i = 0; i < 9; i++) {
			matrix[i] -= other.matrix[i];
		}
		return *this;
	}

	//! return the determinant of the matrix
	FloatType det() const {
		return det3x3();
	}


	//! get the x column out of the matrix
	point3d<FloatType> xcol() const {
		return point3d<FloatType>(matrix[0],matrix[3],matrix[6]);
	}
	//! get the y column out of the matrix
	point3d<FloatType> ycol() const {
		return point3d<FloatType>(matrix[1],matrix[4],matrix[7]);
	}
	//! get the y column out of the matrix
	point3d<FloatType> zcol() const {
		return point3d<FloatType>(matrix[2],matrix[5],matrix[8]);
	}
	//! get the x row out of the matrix
	point3d<FloatType> xrow() const {
		return point3d<FloatType>(matrix[0],matrix[1],matrix[2]);
	}
	//! get the y row out of the matrix
	point3d<FloatType> yrow() const {
		return point3d<FloatType>(matrix[3],matrix[4],matrix[5]);
	}
	//! get the y row out of the matrix
	point3d<FloatType> zrow() const {
		return point3d<FloatType>(matrix[6],matrix[7],matrix[8]);
	}

	//! return the inverse matrix; but does not change the current matrix
	Matrix3x3 getInverse() const {
		FloatType inv[9];

		inv[0] = matrix[4]*matrix[8] - matrix[5]*matrix[7];
		inv[1] = -matrix[1]*matrix[8] + matrix[2]*matrix[7];
		inv[2] = matrix[1]*matrix[5] - matrix[2]*matrix[4];

		inv[3] = -matrix[3]*matrix[8] + matrix[5]*matrix[6];
		inv[4] = matrix[0]*matrix[8] - matrix[2]*matrix[6];
		inv[5] = -matrix[0]*matrix[5] + matrix[2]*matrix[3];

		inv[6] = matrix[3]*matrix[7] - matrix[4]*matrix[6];
		inv[7] = -matrix[0]*matrix[7] + matrix[1]*matrix[6];
		inv[8] = matrix[0]*matrix[4] - matrix[1]*matrix[3];

		FloatType matrixDet = det();
		
		FloatType matrixDetr= (FloatType)1.0 / matrixDet;

		Matrix3x3<FloatType> res;
		for (unsigned int i = 0; i < 9; i++) {
			res.matrix[i] = inv[i] * matrixDetr;
		}
		return res;

	}

	//! overwrite the current matrix with its inverse
	void invert() {
		*this = getInverse();
	}
	//! return the transposed matrix
	Matrix3x3 getTranspose() const {
		Matrix3x3<FloatType> result;
		for(unsigned char x = 0; x < 3; x++) {
			result.at(x,0) = at(0,x);
			result.at(x,1) = at(1,x);
			result.at(x,2) = at(2,x);
		}
		return result;
	}
	//! transpose the matrix in place
	void transpose() {
		*this = getTranspose();
	}

	FloatType* getRawData()
	{
		return matrix;
	}

	//! computes the tensor product between two vectors
	static Matrix3x3 tensorProduct(const point3d<FloatType> &v0, const point3d<FloatType> &v1) {
		Matrix3x3 ret;
		ret._m00 = v0.x * v1.x;		ret._m01 = v0.x * v1.y;		ret._m02 = v0.x * v1.z;
		ret._m10 = v0.y * v1.x;		ret._m11 = v0.y * v1.y;		ret._m12 = v0.y * v1.z;
		ret._m20 = v0.z * v1.x;		ret._m21 = v0.z * v1.y;		ret._m22 = v0.z * v1.z;
		return ret;
	}

	

	//! save matrix array to .matArray
	static void saveMatrixArrayToFile(const std::string& name, const Matrix3x3<FloatType>* matrixArray, unsigned int numMatrices) {
		assert(false);	//UNTESTED
		std::ofstream out(name.c_str());
		out << numMatrices << "\n";
		out << 3 << " " << 3 << "\n";
		for (unsigned int h = 0; h < numMatrices; h++) {
			for (unsigned int i = 0; i < 3; i++) {
				for (unsigned int j = 0; j < 3; j++) {
					out << matrixArray[h](i,j) << " ";
				}
				out << "\n";
			}
		}
	}

	//! load matrix array from .matArray
	static void loadMatrixArrayFromFile(const std::string &name, std::vector<Matrix3x3<FloatType>>& matrices) {
		assert(false);		//UNTESTED
		std::ifstream in(name.c_str());
		unsigned int numMatrices, height, width;;
		in >> numMatrices >> height >> width;
		assert(height == width && height == 3);
		matrices.resize(numMatrices);
		for (unsigned int k = 0; k < numMatrices; k++) {
			Matrix3x3<FloatType> &curr = matrices[k];
			for (unsigned int i = 0; i < 3; i++) {
				for (unsigned int j = 0; j < 3; j++) {
					in >> curr.matrix2[i][j];
				}
			}
		}

	}


	//! save matrix to .mat file
	void saveMatrixToFile(const std::string &name) const {
		std::ofstream out(name.c_str());
		out << 3 << " " << 3 << "\n";
		for (unsigned int i = 0; i < 3; i++) {
			for (unsigned int j = 0; j < 3; j++) {
				out << (*this)(i,j) << " ";
			}
			out << "\n";
		}
	}
	//! load matrix from .mat file
	void loadMatrixFromFile(const std::string &name) {
		std::ifstream in(name.c_str());
		unsigned int height, width;
		in >> height >> width;
		assert(height == width && height == 3);
		for (unsigned int i = 0; i < 3; i++) {
			for (unsigned int j = 0; j < 3; j++) {
				in >> matrix2[i][j];
			}
		}
	}


	//! constructs a matrix from a normal vector (TODO check if it is not transposed...)
	static inline Matrix3x3 frame(const point3d<FloatType>& n) {
		point3d<FloatType> dx0 = point3d<FloatType>(1,0,0) ^ n;
		point3d<FloatType> dx1 = point3d<FloatType>(0,1,0) ^ n;
		point3d<FloatType> dx = (dx0.lengthSq() > dx1.lengthSq() ? dx0 : dx1).getNormalized();
		point3d<FloatType> dy = (n^dx).getNormalized();
		return Matrix3x3(dx,dy,n).getTranspose();
	}

	unsigned int rank(FloatType eps = (FloatType)0.00001) const {
		return util::rank<Matrix3x3<FloatType>, FloatType>(*this, 3, eps);
	}

	EigenSystem<FloatType> eigenSystem() const {
		return EigenSolver<FloatType>::solve<EigenSolver<FloatType>::TYPE_DEFAULT>(*this);
	}


protected:
	//! calculate determinant of a 3x3 sub-matrix given by the indices of the rows and columns
	FloatType det3x3(unsigned int i0 = 0, unsigned int i1 = 1, unsigned int i2 = 2, unsigned int j0 = 0, unsigned int j1 = 1, unsigned int j2 = 2) const {
		return
			  (matrix2[i0][j0]*matrix2[i1][j1]*matrix2[i2][j2])
			+ (matrix2[i0][j1]*matrix2[i1][j2]*matrix2[i2][j0])
			+ (matrix2[i0][j2]*matrix2[i1][j0]*matrix2[i2][j1])
			- (matrix2[i2][j0]*matrix2[i1][j1]*matrix2[i0][j2])
			- (matrix2[i2][j1]*matrix2[i1][j2]*matrix2[i0][j0])
			- (matrix2[i2][j2]*matrix2[i1][j0]*matrix2[i0][j1]);
	}

private:
	union {
		//! access matrix using a single array
		FloatType matrix[9];
		//! access matrix using a two-dimensional array
		FloatType matrix2[3][3];
		//! access matrix using single elements
		struct { 
			FloatType
				_m00, _m01, _m02,
				_m10, _m11, _m12,
				_m20, _m21, _m22;
		};
	};
};

//! writes to a stream
template <class FloatType> 
inline std::ostream& operator<<(std::ostream& s, const Matrix3x3<FloatType>& m)
{ 
	return (
		s << 
		m(0,0) << " " << m(0,1) << " " << m(0,2) << std::endl <<
		m(1,0) << " " << m(1,1) << " " << m(1,2) << std::endl <<
		m(2,0) << " " << m(2,1) << " " << m(2,2) << std::endl
		);
}

//! reads from a stream
template <class FloatType> 
inline std::istream& operator>>(std::istream& s, const Matrix3x3<FloatType>& m)
{ 
	return (
		s >> 
		m(0,0) >> m(0,1) >> m(0,2) >>
		m(1,0) >> m(1,1) >> m(1,2) >>
		m(2,0) >> m(2,1) >> m(2,2)
		);
}


typedef Matrix3x3<int> mat3i;
typedef Matrix3x3<int> mat3u;
typedef Matrix3x3<float> mat3f;
typedef Matrix3x3<double> mat3d;

}  // namespace ml

#endif  // CORE_MATH_MATRIX3X3_H_
