import java.util.NoSuchElementException
val aa = List(1,2,3,4)
def max2(xs: List[Int]): Int = {
 if (xs.isEmpty) 
 {throw new NoSuchElementException()}
 else {
		if (xs.tail.isEmpty)
		{xs.head}
	else
	{
		if (xs.head < max2(xs.tail))
			{max2(xs.tail)}
		else {xs.head}
	}
}
}

def sum2(xs: List[Int]): Int = {
 if (xs.isEmpty) 
 {throw new NoSuchElementException()}
 else {
		if (xs.tail.isEmpty)
		{xs.head}
	else
	{
		xs.head+sum2(xs.tail)
	}
}
}
