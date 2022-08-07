BEGIN { for(i=1;i<=10000;i++) a[i]="A"; k=0; p=0; q=0; }

{  if($1!="$$$$") 
   { k++; 
     a[k]=$0;  
     if(p==1) 
     { b=$0; p=0;} 
     if($2=="<Source_ChemicalName>") p=1; 
   }
   if(q==1)    
   { 
     n=split(b,c," ");
     if(n>1)
     {
      var = c[1];
       for(i=2;i<=n;i++) { var = var "_" c[i] }
       print var;
     }    
     else { print b; }
     for(i=2;i<k;i++) print a[i]; 
     print; 
     k=0; 
     q=0;
   }  
   if($1=="$$$$") { q=1; k++; a[k]=$0; }
}

END{ 
     n=split(b,c," ");
     if(n>1)
     {
      var = c[1];
       for(i=2;i<=n;i++) { var = var "_" c[i] }
       print var;
     }    
     else { print b; }
     for(i=2;i<k;i++) print a[i]; 
     print; 
     k=0; 
     q=0;
   }  
   

