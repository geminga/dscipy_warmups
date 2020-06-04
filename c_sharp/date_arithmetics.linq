<Query Kind="Statements" />

DateTime wantsMaxOfMonth = DateTime.Parse("2020-02-15");
int daysInThisMonth = DateTime.DaysInMonth(wantsMaxOfMonth.Year, wantsMaxOfMonth.Month);
int daysInThisMonth28 = DateTime.DaysInMonth(wantsMaxOfMonth.Year, wantsMaxOfMonth.Month)-27;
// DateTime? wantsMinOfMonth = 1;


DateTime? aDate = DateTime.Now;
DateTime? aDate2 = DateTime.Now;
DateTime? aDate3 = DateTime.Now;
DateTime? aDate4 = DateTime.Now;

DateTime? wants28DaysMinFromMaxOfMonth = null;
DateTime? wants28DaysMaxFromMaxOfMonth  = null;

Console.WriteLine(wantsMaxOfMonth);
Console.WriteLine(daysInThisMonth );
Console.WriteLine(daysInThisMonth28);
