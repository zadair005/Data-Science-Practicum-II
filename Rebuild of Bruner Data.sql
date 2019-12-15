--Rebuild of Bruner Patient Population Dataset
SELECT
	pe.PAT_ID, --Id of the Patient
	pe.PAT_ENC_CSN_ID, --Contact Serial Num of the patient
	pe.PAT_ENC_DATE_REAL,
	pe.CONTACT_DATE, --Date of Contact
	pe.ENC_TYPE_C, --Type of Contact
	pe.DEPARTMENT_ID, --Department where the contact happened
	pe.BP_SYSTOLIC, --Type of blood pressure reading
	pe.BP_DIASTOLIC, --Type of blood pressure reading 2
	pe.TEMPERATURE, --Body temperature of patient
	pe.PULSE, -- heart rate of patient
	pe.WEIGHT, --Patient weight
	pe.HEIGHT, --Patient height
	pe.RESPIRATIONS, --Patients respirations rate
	pe.CHECKIN_TIME, -- Check in time
	pe.CHECKOUT_TIME, -- Check out time
	pe.ENC_CLOSE_DATE, --Date encounter closed
	pe.SERV_AREA_ID,  --Id of service area
	pe.BMI, --Patient BMI
	pe.BSA, --Patient Body Surface Area
	pe2.PHYS_BP, --Patient's Bloodpressure that was entered at the encounter
	pe2.PAT_PAIN_SCORE_C, --Patients pain score
	pe4.FAMILY_SIZE, --# in the patient's family
	pe4.FAMILY_INCOME, --Family income
	pe4.PAT_HOMELESS_YN, --Is the patient homeless?
	pe4.TOBACCO_USE_VRFY_YN, --Does the patient use tobacco? Y = Yes, NULL = ?, N = No
	p.PAT_ID, --Unique Patient ID
	p.PAT_NAME, --Patint Name
	--p.ADD_LINE_1, --Address Line 1
	--p.ADD_LINE_2, --Address Line 2
	p.CITY, --Patient City
	p.STATE_C, --Patient State
	p.COUNTY_C, --Patient County
	zcc.NAME "County_Name", --Patient County Name
	p.ZIP, --Patient Zip Code
	p.BIRTH_DATE, --Patient Birthdate
	p.SEX_C, --Patient Gender 1 = Female, 2 = Male, 3 = Unknown
	p.ETHNIC_GROUP_C, --Patient Race 1 = Non-Hispanic, 2 = Hispanic, 3 = Unknown
	p.MARITAL_STATUS_C, --Patient Marital status 1 = Single, 2 = Married, 3 = Legally Separated, 4 = Divorced, 5 = Widowed, 
	--6 = Unknown, 7 = Significant Other, 8 = Domestic Partner, 9 = Common Law, 10 = Annulled, 11 = Interlocutory, 12 = Never Married,
	--13 = Polygamous
	p.LANGUAGE_C, --Language patient speaks
	p.DEF_FIN_CLASS_C, --Patient Financial Class
	p.CUR_PRIM_LOC_ID, --PCP Location ID
	p.PAT_MRN_ID, --Patient Medical Record Number
	cs.PROV_ID, --Provider ID
	cs.PROV_NAME, --Provider Name
	cs.PROV_TYPE, --Provider Type 0 - Resource 1 - Physician
	cs.ACTIVE_STATUS, --Is the Provider Active?
	cs.STAFF_RESOURCE, --What kind of resource worked with the patient
	cs.EXTERNAL_NAME,
	loc.SERV_AREA_ID, --Service area ID
	loc.LOC_NAME, --Location Name
	had.DX_ID, --List of diagnoses for the appointment 
	edg.CURRENT_ICD10_LIST, --ICD-10 Code
	edg.DX_NAME, -- ICD Description
	had.LINE --Diagnosis Line
FROM PAT_ENC pe --Patient data for each Encounter pt.1
	INNER JOIN PAT_ENC_2 pe2 --Patient data for each Encounter pt.2
		ON pe.PAT_ENC_CSN_ID = pe2.PAT_ENC_CSN_ID
	INNER JOIN PAT_ENC_4 pe4 --Patient data for each Encounter pt.3
		ON pe.PAT_ENC_CSN_ID = pe4.PAT_ENC_CSN_ID
	INNER JOIN PATIENT p --Patient demographic information
		ON pe.PAT_ID = p.PAT_ID
		AND pe.PAT_ENC_CSN_ID = p.MEDS_LAST_REV_CSN
	INNER JOIN CLARITY_LOC loc --Get's us the Bruner Locations
		ON pe.SERV_AREA_ID = loc.SERV_AREA_ID
	INNER JOIN HSP_ADMIT_DIAG had --Connect ICD to Patient
		ON had.PAT_ENC_CSN_ID = pe.PAT_ENC_CSN_ID
		AND had.PAT_ID = pe.PAT_ID
		AND had.PAT_ENC_DATE_REAL = pe.PAT_ENC_DATE_REAL
	INNER JOIN CLARITY_EDG edg --Get's us the ICD codes & descriptions
		ON edg.DX_ID = had.DX_ID
	LEFT OUTER JOIN ZC_COUNTY zcc --Gives us the actual county name and not just a numbered code
		ON zcc.COUNTY_C = p.COUNTY_C
	INNER  JOIN CLARITY_SER cs --Providers Information
		ON p.CUR_PCP_PROV_ID = cs.PROV_ID
WHERE pe.CONTACT_DATE >= '2017-01-01' --Only patients that visited since the start of 2017
	AND pe.SERV_AREA_ID = '2' --The service area for all the of the Bruner Clinic patients
	AND loc.LOC_NAME LIKE '%Bruner%' --Will restrict to strictly Bruner Clinic patients
	AND edg.CURRENT_ICD10_LIST IN ('I10','I15.9','I15.1','I15.1','I15.0','O10.919', 'G93.2', 'I12.9','I27.20','I27.21','I27.21, K76.6','I27.24',
		'I87.309', 'K76.6', 'E03.9', 'E78.5',--Hypertension Codes
		'E11.9','E11.65','E11.29','E11.8','E11.42','E11.69','E11.21','E11.22','E11.40','E11.319','E11.59','E10.9','E11.41','E13.9','E11.3293',
		'E13.8','E11.311', 'E08.621, L97.509', 'E10.10','E10.22, N18.5', 'E10.3511','E10.35.22','E10.621, L97.529', 'E10.649','E10.9','E11.10',
		'E11.22, N18.5, Z79.4','E11.3522','E11.3591', 'E11.42','E11.40, G99.2','E11.51','E11.52','E11.610','E11.621, L97.509','E11.8','E11.9',
		'E11.9, Z79.4','E13.10','E13.22, N18.5, E13.65', 'E13.621, L97.505','O24.113','O24.414','O24.419', 'O24.419, Z79.4', --Diabetes Codes
		'E66.9','E66.01','O99.212','O99.210','O99.214','O99.213') -- Obesity Codes 
ORDER BY cs.PROV_NAME
