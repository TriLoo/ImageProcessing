/*
 * AxiTimerHelper.h
 *
 *  Created on: 06/07/2015
 *      Author: laraujo
 */

#ifndef AXITIMERHELPER_H_
#define AXITIMERHELPER_H_

#include "xil_types.h"
#include "xtmrctr.h"
#include "xparameters.h"

class AxiTimerHelper {
public:
	AxiTimerHelper();
	virtual ~AxiTimerHelper();
	unsigned int getElapsedTicks();
	double getElapsedTimerInSeconds();
	unsigned int startTimer();
	unsigned int stopTimer();
	double getClockPeriod();
	double getTimerClockFreq();
private:
	XTmrCtr m_AxiTimer;
	unsigned int m_tickCounter1;
	unsigned int m_tickCounter2;
	double m_clockPeriodSeconds;
	double m_timerClockFreq;
};

#endif /* AXITIMERHELPER_H_ */
