/* Creating the database */
create database bank_payam

use bank_payam

/* Creating the table in the database */

/*Table 1 */
create table UserLogins (UserLoginID smallint primary key not null, 
						UserLogin char(15) not null, UserPassword varchar(20) not null)


/*Table 2 */
create table UserSecurityQuestions (UserSecurityQuestionID tinyint primary key not null,
						UserSecurityQuestion varchar(50) not null)

/*Table 3 */
create table SavingsInterestRates (InterestSavingsRateID tinyint primary key not null,
								InterestRateValue numeric(18,9) not null, 
								InterestRateDescription varchar(20))

/*Table 4 */
create table AccountType (AccountTypeID tinyint primary key not null, 
							AccountTypeDescription varchar(30))

/*Table 5 */
create table TransactionType (TransactionTypeID tinyint primary key not null,
							TransactionTypeName char(10) not null,
							TransactionTypeDescription varchar(50),
							TransactionFeeAmount smallmoney not null)

/*Table 6 */
create table AccountStatusType (AccountStatusTypeID tinyint primary key not null,
							 AccountStatusDescription varchar(30))


/*Table 7 */
create table Employee (EmployeeID int primary key not null, EmployeeFirstName varchar(25) not null, 
						EmployeeMiddleInitial char(1), EmployeeLastName varchar(25) not null,
						EmployeeIsManager bit not null)  


/*Table 8 */
create table Account (AccountID int primary key not null, CurrentBalance int not null, 
					AccountTypeID tinyint foreign key references AccountType(AccountTypeID) not null, 
					AccountStatusTypeID tinyint foreign key 
					references AccountStatusType(AccountStatusTypeID) not null, 
					InterestSavingsRateID tinyint foreign key 
					references SavingsInterestRates(InterestSavingsRateID))


/*Table 9 */
create table OverDraftLog (AccountID int primary key not null foreign key references Account(AccountID),
						  OverDraftDate datetime not null , OverDraftAmount money not null, 
						  OverDraftTransactionXML varchar(30))


/*Table 10 */
create table Customer (CustomerID int primary key not null, 
						AccountID int foreign key references Account(AccountID) not null, 
						CustomerAddress1 varchar(30) not null, CustomerAddress2 varchar(30), 
						CustomerFirstName varchar(30) not null, CustomerMiddleInitial char(1), 
						CustomerLastName varchar(30) not null, City varchar(20) not null, 
						StateName char(2) not null, ZipCode char(10) not null, EmailAddress varchar(40),
						HomePhone char(10) not null, CellPhone char(10), WorkPhone char(10), SSN char(9) 
						not null, 
						UserLoginID smallint foreign key references UserLogins(UserLoginID) not null)


/*Table 11 */
create table CustomerAccount (AccountID int not null foreign key references Account(AccountID), 
							CustomerID int foreign key references Customer(CustomerID))


/*Table 12 */
create table TransactionLog (TransactionID int primary key not null, TransactionDate datetime not null, 
							TransactionTypeID tinyint foreign key
							references TransactionType(TransactionTypeID) not null, 
							TransactionAmount money not null, NewBalance money not null, 
							AccountID int not null foreign key references Account(AccountID), 
							CustomerID int not null foreign key references Customer(CustomerID), 
							EmployeeID int not null foreign key references Employee(EmployeeID), 
							UserLoginID smallint not null foreign key references UserLogins(UserLoginID))


/*Table 13 */
create table UserSecurityAnswers (UserLoginID smallint primary key not null foreign key 
								references UserLogins(UserLoginID), 
								UserSecurityAnswer varchar(25) not null, 
								UserSecurityQuestionID tinyint foreign key 
								references UserSecurityQuestions(UserSecurityQuestionID) not null)


/*Table 14 */
create table LoginAccount (UserLoginID smallint foreign key 
                           references UserLogins(UserLoginID) not null, 
							AccountID int foreign key references Account(AccountID) not null)


/*Table 15 */
create table FailedTransactionErrorType (FailedTransactionErrorTypeID tinyint primary key not null, 
										FailedTransactionDescription varchar(50))


/*Table 16 */
create table FailedTransactionLog (FailedTransactionID int primary key not null, 
								FailedTransactionErrorTypeID tinyint foreign key 
								references FailedTransactionErrorType(FailedTransactionErrorTypeID) 
								not null, 
								FailedTransactionErrorTime datetime not null, 
								FailedTransactionXML varchar(30))


/*Table 17 */
create table LoginErrorLog (ErrorLogID int primary key not null, 
							ErrorTime datetime not null, FailedTransactionXML varchar(30))



/* Inserting Values into the tables */

/*Values: Table 1 */
insert into UserLogins values(12341, 'mwhite','nrasooli30'),(12342, 'vangogh', 'pbagheri30'),
						     (12343, 'lilianf','snazeri30'),(12344, 'hshrine','sbrown30'),
							 (12345, 'john_jackson','kwhite30')


/*Values: Table 2 */
insert into UserSecurityQuestions values(231, 'what''s you pet''s name?'), 
										(232, 'what''s your mother''s middle name?'),
										(233, 'what''s your middle name?'), 
										(234, 'Who was your chilhood hero?'),
										(235, 'Where did you spend your honey moon?')


/*Values: Table 3 */
insert into SavingsInterestRates values(31, 2.34, 'IFSA'), (32, 1.45, 'RRSP'),
								(33, 3.23,'Mutual Fund'), (34, 3.75, 'Low Risk Mutual Fund'),
								(35, 1.75, 'Savings Account')

/*Values: Table 4 */
insert into AccountType values(41, 'Checking'),(42, 'Saving')


/*Values: Table 5 */
insert into TransactionType values(51, 'Interac','Internet banking', 3.00),
								  (52, 'Pay Bill', 'Internet banking', 2.00),
								  (53, 'ChekToSav', 'Internal transfer', 0.00),
								  (54, 'SavToChek', 'Internal transfer', 0.50),
								  (55, 'SavToCred', 'Internal transfer', 0.00)
			

/*Values: Table 6 */			
insert into AccountStatusType values(61, 'Active'),(62, 'Inactive'),(63, 'Temporary'),
									(64, 'Requested'),(65, 'OnHold')

						
/*Values: Table 7 */
insert into Employee values(7231, 'Albert', 'B', 'Einstein', 1),(7232, 'Isaac', 'H', 'Newton', 1),
							(7233, 'Ernst', 'C', 'Mach', 0),(7234, 'David', 'C', 'Deutsch', 0),
							(7235, 'Karl', 'M', 'Reimann', 1)


/*Values: Table 8 */
insert into Account values(81231, 5600, 42, 61, 35),(81232, 3300, 41, 62, null), 
						  (81233, 500, 41, 64, null), (81234, 7200, 42, 63, 35),
						  (81235, 400, 42, 65, 35)


/* Values: Table 9 */
insert into OverDraftLog values (81233, '2002-11-15 17:22:31', 230, 'xml91'),
								(81232, '2010-05-14 19:01:02', 40, 'xml92'),
								(81234, '2006-01-24 12:34:22', 120, 'xml93'), 
								(81231, '2002-03-04 11:27:15', 38, 'xml94'),
								(81235, '1999-10-17 09:12:47', 100, Null)


/*Values: Table 10 */
insert into Customer values (10121, 81234, '191 Jarvis St.', null, 'Vincent', null, 'Van Gogh', 
							'Toronto', 'ON', 'M5A 7P1', 'vinc@gmail.com', '4168856756', null, null,
							'135545243', 12342),
							(10122, 81233, '34 Elizabeth St.', null, 'Mark', 'H', 'White', 
							'London', 'ON', 'M1B 3S7', 'mark@gmail.com', '4164563421', null, 
							'7095567187', '975565431', 12341),
							(10123, 81231, '45 Victoria Ave.', 'East side', 'Holy', 'K', 'Shrine', 
							'Vancouver', 'BC', 'Z2R 4FT', 'holysh@hotmail.com', '4258563921', null, 
							 null, '134567324', 12344),
							 (10124, 81235, '252 Yonge St.', null, 'John', 'S', 'Jackson', 
							'Toronto', 'ON', 'M4A 3X1', 'johnsj@gmail.com', '6478763111', null, 
							'4164542542', '934567233', 12345),
							(10125, 81232, '121 Queen St.', null, 'Lilian', null, 'Florin', 
							'Calgary', 'AB', 'G5R 5F4', 'lili234@gmail.com', '5268765643', null, 
							null, '466425776', 12343)

/*Values: Table 11 */
insert into CustomerAccount values (81233, 10122), (81234, 10121), (81231, 10123), 
								   (81235, 10124), (81232, 10125)

/*Values: Table 12 */
insert into TransactionLog values (12121, '2001-09-23 09:23:45', 51, 765.45, 1232.55, 81235, 10124, 
								  7231, 12345),
								  (12122, '2004-02-14 11:15:34', 53, 345.02, 986.34, 81234, 10121, 
								  7233, 12342),
								  (12123, '2014-12-24 17:33:54', 52, 445.56, 1986.44, 81233, 10122, 
								  7235, 12341),
								  (12124, '2015-11-04 19:03:25', 55, 745.59, 686.24, 81232, 10125, 
								  7232, 12343),
								  (12125, '2000-10-10 09:33:28', 55, 75.69, 5686.64, 81231, 10123, 
								  7234, 12344)

/*Values: Table 13 */
insert into UserSecurityAnswers values (12341, 'pashmak', 231), (12342, 'silver', 231), 
									   (12343, 'rostam', 234), (12344, 'Shahsavar', 235),
									   (12345, 'spiderman', 234)

/*Values: Table 14 */
insert into LoginAccount values (12344, 81231), (12343, 81232), (12341, 81233), 
								(12342, 81234), (12345, 81235)


/*Values: Table 15 */
insert into FailedTransactionErrorType values (151, 'Account empty'), (152, 'Overdraft not allowed'),
											  (153, 'Request exceeded existing balance'),
											  (154, 'Coonection lost'), (155, 'Please try again')


/*Values: Table 16 */
insert into FailedTransactionLog values (16121, 152, '2004-12-14 11:15:34', 'xml1'),
										(16122, 151, '2014-09-14 12:25:34', 'xml2'),
										(16123, 154, '2006-12-04 09:15:33', 'xml3'),
										(16124, 151, '2012-07-23 05:08:56', 'xml4'),
										(16125, 155, '2007-02-04 21:12:10', 'xml5')

/*Values: Table 17 */
insert into LoginErrorLog values (17121, '2015-10-23 10:45:17', 'xml1'),
								 (17122, '2011-02-17 08:42:51', null),
								 (17123, '2000-07-15 10:45:18', 'xml3'),
								 (17124, '2015-09-13 10:45:17', null),
								 (17125, '2001-11-11 11:15:27', 'xml5')




/* Creating views */

-- A view that shows customers with checking account
create view WhoHasChecking as
select c.CustomerFirstName, c.CustomerMiddleInitial ,c.CustomerLastName 
from Customer c, Account a
where c.AccountID = a.AccountID and a.AccountTypeID = 41 and c.StateName = 'ON'

select * from WhoHasChecking



-- A view that shows customers with balance > 5000
/* For this question I havent't included the interest rate because we didnt have
the time interval that the money has been in the account. */
create view BalanaceMoreThan5000 as
select c.CustomerFirstName, c.CustomerMiddleInitial ,c.CustomerLastName
from Customer c, Account a
where c.AccountID = a.AccountID and a.currentbalance > 5000

select * from BalanaceMoreThan5000



-- A view that shows the number of customers with either checking or saving account
create view NumberOfEachAccountType as
select t.AccountTypeDescription, count(a.AccountTypeID) AccoutCount
from Account a, AccountType t
where a.AccountTypeId = t.AccountTypeID
group by t.AccountTypeDescription

select * from NumberOfEachAccountType



-- A view that shows username and password for any particular AccountID (user)
create view UserNamePassword as
select c.AccountID , u.UserLogin, u.UserPassword
from Customer c, LoginAccount l, UserLogins u
where c.AccountID = l.AccountID and l.UserLoginID = u.UserLoginID

select * from UserNamePassword



-- A view that shows all customers' overdraft amount
create view OverDrafts as
select c.CustomerFirstName, c.CustomerMiddleInitial, c.CustomerLastName, o.OverDraftAmount
from Customer c, OverDraftLog o
where c.AccountID = o.AccountID

select * from OverDrafts






