#pragma once


/************************************************************************/
/* Reads binary dump data from .sensor files                            */
/************************************************************************/

#include "GlobalAppState.h"
#include "RGBDSensor.h"
#include "stdafx.h"

#ifdef BINARY_DUMP_READER

class BinaryDumpReader	: public RGBDSensor
{
public:

	//! Constructor
	BinaryDumpReader();

	//! Destructor; releases allocated ressources
	~BinaryDumpReader();

	//! initializes the sensor
	HRESULT createFirstConnected();

	//! reads the next depth frame
	HRESULT processDepth();
	

	HRESULT processColor()	{
		//everything done in process depth since order is relevant (color must be read first)
		return S_OK;
	}

	std::string getSensorName() const {
		//return "BinaryDumpReader";
		return m_data.m_SensorName;
	}

	mat4f getRigidTransform(int offset) const {
		unsigned int idx = m_CurrFrame - 1 + offset;
		if (idx >= m_data.m_trajectory.size()) throw MLIB_EXCEPTION("invalid trajectory index " + std::to_string(idx));
		return m_data.m_trajectory[idx];
	}

private:
	//! deletes all allocated data
	void releaseData();

	CalibratedSensorData m_data;

	unsigned int	m_NumFrames;
	unsigned int	m_CurrFrame;
	bool			m_bHasColorData;

};


#endif
