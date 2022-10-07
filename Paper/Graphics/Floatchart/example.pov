// Ausführen: povray settings minimal.pov
// povray "settings[high]" minimal.pov
// povray settings +W100 minimal.pov

#version 3.7;
// Einbinden von Bibliotheken
#include "colors.inc"
#include "shapes.inc"
#include "transforms.inc"
#include "functions.inc"
#include "textures.inc"
include "metals.inc"  



// Globale Variablen, auch mit ifdef, bool (on,off)
#declare Use_FocalBlur=off;  
#declare TimeEv = on;
// Declare variables
#ifndef (Background_Finish)
	#declare Background_Finish = finish{ambient 0.2 diffuse 0.8 brilliance 1.0}
#end 




// Global Settings
global_settings { // Beleuchtung global anpassen
	//assumed_gamma 1.1
	//ambient_light <0.0,0.0,0.5>
	//ambient_light 0.1
}   


// Standardwerte, falls nicht von Objekt überschrieben
#default{finish{ambient 0.0				// Helligkeit aus indirekter Beleuchtung
				diffuse 0.1			// Helligkeit aus direkter Beleuchtung
				phong 0.0 				// Glanzlichter
				specular 0.05			// flächiger Schimmer
				reflection 0.0 		// Helligkeit aus Spiegelung
				brilliance 2.0		// Abfall der Helligkeit abhängig vom Einfallswinkel
				//crand 0.3 				// Körnigkeit
				}}

background{color White}



// Kamera: Position, Ausrichtung, Zoom, ...
#declare cam_r = 200;
#declare cam_phi = 0;
#declare cam_theta = 0;
#declare cam_angle = 1.5;
#declare cam_sky = <0, 0, 1>;
#declare cam_rot = <0, 0, 0>;
#declare cam_right = <-5/4, 0, 0>;
#declare cam_loc = <cam_r*sin(cam_theta)*cos(cam_phi), cam_r*sin(cam_theta)*sin(cam_phi), cam_r*cos(cam_theta)>;
camera{
	perspective	// Projektion
	location cam_loc	// Position
	angle cam_angle		// Winkel (Lochkamera)
	rotate cam_rot		// Rotation
	sky cam_sky				// Oben
	right cam_right		// Rechts (standardmäßig linkshändiges Koordinatensystem)
	look_at -cam_loc	// Ausrichtung der Kamera
	#if (Use_FocalBlur) // Tiefenschärfe (sehr Rechenintensiv!)
		focal_point  <0, 0, 0>
		aperture 2
		blur_samples 512
		confidence 0.95
		variance 1./4096
  #end
}






// Lichtquellen (mindestens eine, Punktlichter, Fl�chenlichter, ...)
#declare light_r = 200;
#declare light_phi = 0;
#declare light_theta = 0;
#declare light_color = <1, 1, 1>;

#declare light_loc = <light_r*sin(light_theta)*cos(light_phi), light_r*sin(light_theta)*sin(light_phi), light_r*cos(light_theta)>;
light_source{
	light_loc
	color rgb light_color
	area_light <10, 0, 0>, <0, 0, 01>, 8, 8 // Flächige Lichtquelle für weiche Schatten
}


#declare light_loc = <cam_r*sin(cam_theta)*cos(cam_phi), cam_r*sin(cam_theta)*sin(cam_phi), cam_r*cos(cam_theta)>;
light_source{
	light_loc
	color rgb light_color
}  

light_source
{
  <-2, -1, 19>
  color White*1.5 // White*0.8 
  area_light <1,0,0>*4.5, <-0,1,0>*4.5, 3, 3
  jitter
  spotlight
  radius 3.2
  falloff 13.2
   tightness 10
  point_at <0, 0, 0>
} 

#declare lattice_finish = finish
{
  ambient    0.1
  diffuse    1
  reflection {0.}
  phong      0.2
  brilliance 2
}


// Eigene Macros (Funktionen)----------------------------------------------- 
#macro Atom(pos)
object{
	    sphere{pos, 0.07}
	    texture{
		    pigment{color rgbt <1,1,1, 0.7>}
		    finish{brilliance 0.1
		           reflection 0
		           phong 0.1} }
	    } 
	  
#end
     
     
  




#macro Arrow(Base, Vector, TipRatio, Radius,acolor) 
union{
cylinder{
	Base, Base + (1.0-TipRatio)*Vector, Radius
}
cone{
	Base+(1.0-TipRatio)*Vector, 2*Radius
	Base+Vector, 0.0
}
	sphere{
		Base, Radius }
texture{pigment{acolor}
finish{lattice_finish ambient 0.6}}       }
#end


#macro Arrow_mounted(Center, Vector, TipRatio, Radius, Mount, Color)
#local Base = - Mount* Vector;
union{
	Arrow(Base, Vector, TipRatio, Radius,Color)
	//no_shadow
	texture{
		pigment{ Color }
		finish{brilliance 1
		diffuse 1
		ambient 0.2
		crand 0.}}
	rotate 4*y 
	rotate -4*z
	translate Center
}
#end
    





//--------------------------------------------------------------------------------------
      

union{ 
    object{
	    sphere_sweep{linear_spline 2, 
	    <-2.15,0,0>,0.03,
	     <2.15,0,0>,0.03}
	    texture{ pigment{color rgbt <1,1,1, 0.7>}
	    finish {reflection 0
	            brilliance 0
	            ambient 0.1
	            diffuse 0.1
	            phong 0
	            specular 0
	            metallic 0}}
}
  
#declare Rnd_1 = seed (1153);        
#declare Rnd_2 = seed (13);   
#for(n,-2.1,2.1,0.7)
Atom(<n,0,0>)  

#declare r = rand(Rnd_1)*0.4;

#declare sign = rand(Rnd_2) ;
#if(sign>0.5)
#declare sign=1;
#else
#declare sign=-1;
#end
#declare dir=<r, sign*sqrt(0.45*1.5-r*r),0>*1.6; 
Arrow_mounted(<n,0,0.>, dir, 0.2,0.045,0.4, color rgb <1,0,0>)
#end  

scale 1.0
}
  
  
