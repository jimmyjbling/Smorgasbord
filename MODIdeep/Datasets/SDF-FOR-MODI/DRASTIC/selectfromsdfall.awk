BEGIN { 
          str = "wc " fileinp " > temp.txt";
          system(str);
          getline a < "temp.txt";
          split(a,b);
          NComps=b[1]; 
#          print NComps;  
          for(i=1;i<=NComps;i++) 
          { getline a < fileinp;
            split(a,b);
            name[i]=b[1];
#            print i,name[i],"\""name[i]"\"";  
          } 
          p=0;
          k=0;
          i=1;
          close(fileinp);
      }

{ 
  if(p==0 && (name[i]==$1 || name[i] == "\""$1"\"")) 
  { 
    p=1; 
#    k++; 
    if(i==1) print name[i];
    if(i>1) { print ""; print name[i]; }  
    i++;
    k=1; 
  }
  if(p==1) 
  {
    if(i>1 && k>1) print;
    k++;
    if($1=="$$$$") { p=0; k=0;}
  }
  if(i>NComps+1) exit; 
}
               