import math
def main():
  print("enter stop to end the program")
  command = ""
  while command != "stop":
     nth_term = int(input("nth term: "))
     sum = summation(nth_term)
     print("partial sum is: ", sum)
def summation(nth_term):
    n = 1
    sum = 0 
    while n <= nth_term:
        sequence = (math.log(n)/n)**1.1
        sum += sequence
        n += 1
    return sum 
def factorial(n):
   result = 1
   while n > 0:
      result *= n
      n -= 1
   return result

# Add this at the end of the script
if __name__ == "__main__":
    main()