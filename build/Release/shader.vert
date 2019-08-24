#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNorm;

uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

out float fragIntensity;

void main()
{
	gl_Position = P*V*M*vec4(aPos, 1.0);
	vec3 epnorm = vec3(V*M * vec4(aNorm,1.0));

	vec3 vertex_pos = vec3(V*M * vec4(aPos,1.0));
	vec3 vertex_vec = normalize(vec3(0,0,0) - vertex_pos);
       
	fragIntensity = (dot(normalize(epnorm), normalize(vertex_vec))+0.5)/1.5;
   
}