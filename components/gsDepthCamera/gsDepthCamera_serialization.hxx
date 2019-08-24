#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>

#include <boost/serialization/split_free.hpp>
#include <boost/serialization/map.hpp>


BOOST_SERIALIZATION_SPLIT_FREE(::libfreenect2::Frame)

namespace boost {
	namespace serialization {

		template<class Archive>
		void save(Archive & ar, const ::libfreenect2::Frame &f, const unsigned int version)
		{
			size_t width = f.width;
			size_t height = f.height;
			size_t bytes_per_pixel = f.bytes_per_pixel;
			ar &width; 
			ar &height;
			ar &bytes_per_pixel;

			ar &f.timestamp; 
			ar &f.received_timestamp;
			ar &f.sequence; 
			ar &f.exposure;
			ar &f.gain; 
			ar &f.gamma; 
			ar &f.status; 
			ar &f.format;
		}

		template<class Archive>
		inline void load_construct_data(Archive & ar, ::libfreenect2::Frame *f, const unsigned int version)
		{
			size_t width, height, bytes_per_pixel;
			ar &width; 
			ar &height; 
			ar &bytes_per_pixel;
			::new(f)libfreenect2::Frame(width, height, bytes_per_pixel);

		}
		
		template<class Archive>
		void load(Archive & ar, ::libfreenect2::Frame &f, const unsigned int version)
		{
			ar &f.timestamp;   
			ar &f.received_timestamp; 
			ar &f.sequence;  
			ar &f.exposure;         
			ar &f.gain;             
			ar &f.gamma;           
			ar &f.status;        
			ar &f.format;         
		}

		template<class Archive>
		void serialize(Archive & ar, libfreenect2::Freenect2Device::IrCameraParams& p, const unsigned int version)
		{
			ar &p.fx; 
			ar &p.fy; 
			ar &p.cx; 
			ar &p.cy; 
			ar &p.k1; 
			ar &p.k2; 
			ar &p.k3; 
			ar &p.p1; 
			ar &p.p2; 
		}

		template<class Archive>
		void serialize(Archive & ar, libfreenect2::Freenect2Device::ColorCameraParams & p, const unsigned int version)
		{
			ar &p.fx; 
			ar &p.fy; 
			ar &p.cx; 
			ar &p.cy; 
			ar &p.shift_d;
			ar &p.shift_m;


			ar &p.mx_x3y0; 
			ar &p.mx_x0y3; 
			ar &p.mx_x2y1; 
			ar &p.mx_x1y2; 
			ar &p.mx_x2y0; 
			ar &p.mx_x0y2; 
			ar &p.mx_x1y1; 
			ar &p.mx_x1y0; 
			ar &p.mx_x0y1; 
			ar &p.mx_x0y0; 

			ar &p.my_x3y0; 
			ar &p.my_x0y3; 
			ar &p.my_x2y1; 
			ar &p.my_x1y2; 
			ar &p.my_x2y0; 
			ar &p.my_x0y2; 
			ar &p.my_x1y1; 
			ar &p.my_x1y0; 
			ar &p.my_x0y1; 
			ar &p.my_x0y0; 
		}

	} // namespace serialization
} // namespace boost