/*
 * AxiTimerHelper.cpp
 *
 *  Created on: 06/07/2015
 *      Author: laraujo
 */

#include "AxiTimerHelper.h"

AxiTimerHelper::AxiTimerHelper() {
	// Initialize the timer hardware...
	 XTmrCtr_Initialize(&m_AxiTimer, XPAR_TMRCTR_0_DEVICE_ID);

	 //XTmrCtr_SetOptions(&m_AxiTimer, 0, XTC_AUTO_RELOAD_OPTION);

	 // Get the clock period in seconds
	 m_timerClockFreq = (double) XPAR_AXI_TIMER_0_CLOCK_FREQ_HZ;
	 m_clockPeriodSeconds = (double)1/m_timerClockFreq;

}

AxiTimerHelper::~AxiTimerHelper() {
	// TODO Auto-generated destructor stub
}

unsigned int AxiTimerHelper::getElapsedTicks() {
	return m_tickCounter2 - m_tickCounter1;
}

unsigned int AxiTimerHelper::startTimer() {
	// Start timer 0 (There are two, but depends how you configured in vivado)
	XTmrCtr_Reset(&m_AxiTimer,0);
	m_tickCounter1 =  XTmrCtr_GetValue(&m_AxiTimer, 0);
	XTmrCtr_Start(&m_AxiTimer, 0);
	return m_tickCounter1;
}

unsigned int AxiTimerHelper::stopTimer() {
	XTmrCtr_Stop(&m_AxiTimer, 0);
	m_tickCounter2 =  XTmrCtr_GetValue(&m_AxiTimer, 0);
	return m_tickCounter2 - m_tickCounter1;
}

double AxiTimerHelper::getElapsedTimerInSeconds() {
	double elapsedTimeInSeconds = (double)(m_tickCounter2 - m_tickCounter1) * m_clockPeriodSeconds;
	return elapsedTimeInSeconds;
}

double AxiTimerHelper::getClockPeriod() {
	return m_clockPeriodSeconds;
}

double AxiTimerHelper::getTimerClockFreq() {
	return m_timerClockFreq;
}
