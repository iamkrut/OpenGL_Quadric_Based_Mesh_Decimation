/*
Sample code by Wallace Lira < http://www.sfu.ca/~wpintoli/> based on
the four Nanogui examples and also on the sample code provided in
https ://github.com/darrenmothersele/nanogui-test

All rights reserved.Use of this source code is governed by a
BSD - style license that can be found in the LICENSE.txt file.
*/

#include <nanogui/opengl.h>
#include <nanogui/glutil.h>
#include <nanogui/screen.h>
#include <nanogui/window.h>
#include <nanogui/layout.h>
#include <nanogui/label.h>
#include <nanogui/checkbox.h>
#include <nanogui/button.h>
#include <nanogui/toolbutton.h>
#include <nanogui/popupbutton.h>
#include <nanogui/combobox.h>
#include <nanogui/progressbar.h>
#include <nanogui/entypo.h>
#include <nanogui/messagedialog.h>
#include <nanogui/textbox.h>
#include <nanogui/slider.h>
#include <nanogui/imagepanel.h>
#include <nanogui/imageview.h>
#include <nanogui/vscrollpanel.h>
#include <nanogui/colorwheel.h>
#include <nanogui/graph.h>
#include <nanogui/tabwidget.h>
#include <nanogui/glcanvas.h>
#include <iostream>
#include <string>
#include <typeinfo>
#include <fstream>
#include <chrono>  // for high_resolution_clock
#include <algorithm>

// Includes for the GLTexture class.
#include <cstdint>
#include <memory>
#include <utility>


#if defined(__GNUC__)
# pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#if defined(_WIN32)
# pragma warning(push)
# pragma warning(disable: 4457 4456 4005 4312)
#endif

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#if defined(_WIN32)
# pragma warning(pop)
#endif
#if defined(_WIN32)
# if defined(APIENTRY)
# undef APIENTRY
# endif
# include <windows.h>
#endif

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;
using std::pair;
using std::to_string;

using nanogui::Screen;
using nanogui::Window;
using nanogui::GroupLayout;
using nanogui::Button;
using nanogui::CheckBox;
using nanogui::Vector2f;
using nanogui::Vector2i;
using nanogui::MatrixXu;
using nanogui::MatrixXf;
using nanogui::Label;
using nanogui::Arcball;
using nanogui::Vector3f;

struct Vertex;
struct Face;
struct W_edge;
      
struct Vertex {
   float x, y, z;
   W_edge *edge;
   bool decimated;
   bool is_new;
   MatrixXf quadric_matrix;

   inline bool operator==(Vertex v) {
      if (v.x==x && v.y== y && v.z == z && v.edge == edge)
         return true;
      else
         return false;
   }
};

std::ostream& operator<<(std::ostream& os, const Vertex& v) {
   return os << "x: " <<  v.x << ", y: " << v.y << ", z: " << v.z; 
}

struct W_edge
{
   Vertex *start, *end;
   Face *left, *right;
   W_edge *left_prev, *left_next, *right_prev, *right_next;
   bool decimated;
   
   inline bool operator==(W_edge w) {
      if (w.start==start && w.end== end)
         return true;
      else
         return false;
   }
};

std::ostream& operator<<(std::ostream& os, const W_edge& edge) {
   return os << "\nstart: " <<  *edge.start << "\nend: " << *edge.end; 
}

struct Face
{
   W_edge *edge;
   bool decimated;
   inline bool operator==(Face f) {
      if (f.edge==edge)
         return true;
      else
         return false;
   }
};

class MyGLCanvas : public nanogui::GLCanvas {
   public:
   MyGLCanvas(Widget *parent) : nanogui::GLCanvas(parent) {
      using namespace nanogui;

      mShader.initFromFiles("a_smooth_shader", "StandardShading.vertexshader", "StandardShading.fragmentshader");

      // After binding the shader to the current context we can send data to opengl that will be handled
      // by the vertex shader and then by the fragment shader, in that order.
      // if you want to know more about modern opengl pipeline take a look at this link
      // https://www.khronos.org/opengl/wiki/Rendering_Pipeline_Overview
      mShader.bind();

      //mShader.uploadIndices(indices);
      mShader.uploadAttrib("vertexPosition_modelspace", positions);
      mShader.uploadAttrib("color", colors);
      mShader.uploadAttrib("vertexNormal_modelspace", normals);

      // ViewMatrixID
      // change your rotation to work on the camera instead of rotating the entire world with the MVP matrix
      Matrix4f V;
      V.setIdentity();
      //V = lookAt(Vector3f(0,0,-5), Vector3f(-5,0,0), Vector3f(0,1,0));
      mShader.setUniform("V", V);

      //ModelMatrixID
      Matrix4f M;
      M.setIdentity();
      mShader.setUniform("M", M);

      // This the light origin position in your environment, which is totally arbitrary
      // however it is better if it is behind the observer
      mShader.setUniform("LightPosition_worldspace", Vector3f(-2, 6, -4));

   }

   //flush data on call
   ~MyGLCanvas() {
      mShader.free();
   }

   //Translation on axes
   void updateTranslationValues(float val, int axis){
      if (axis == 0){
         translationValues[0] = val;
      }
      else if (axis == 1){
         translationValues[1] = val;
      }
   }

   //Rotation on axes
   void updateRotationValues(nanogui::Vector3f vRotation) {
      nanogui::Vector3f sumRotations = mRotation + vRotation;
      mRotation = sumRotations;
   }

   void updateMeshColors(MatrixXf positions, MatrixXf wireframe_positions){

        MatrixXf mesh_colors = MatrixXf(3, positions.size() / 3);
        MatrixXf wireframe_colors = MatrixXf(3, wireframe_positions.size() / 3);
        MatrixXf final_colors = MatrixXf(3, mesh_colors.size() / 3 + wireframe_colors.size() / 3 );

        for (int i = 0; i < positions.size() / 3; i++){
            mesh_colors.col(i) << 1, 0, 0;
        }

        for (int i = 0; i < wireframe_positions.size() / 3; i++){
            wireframe_colors.col(i) << 0, 0, 0;
        }

        final_colors << mesh_colors, wireframe_colors;

        colors = final_colors;
    }

   //Method to update the mesh itself, can change the size of it dynamically, as shown later
   void updateMeshPositions(MatrixXf newPositions, MatrixXf n_wireframe_positions) {
      mesh_positions = newPositions;
      wireframe_positions = n_wireframe_positions;

      MatrixXf temp_positions = MatrixXf(3, mesh_positions.size() / 3 + wireframe_positions.size() / 3 );
      temp_positions << mesh_positions, wireframe_positions;
      positions = temp_positions;
   }

    /////////////////////////*********?///////////////

   //Scaling
   void updateScaleValues(float value){
      scaleValue = value;
   }

   //ComboBox
   void updateMeshRepresentVal(int value) {
      meshRepresentVal = value;
   }

   //Colors 
   void updateMeshColors (MatrixXf newColors){
        colors = newColors;
   }

   void saveMeshtoObj(string fileName, vector<Vector3f> &temp_vertices, vector<Vector3f> &vertexIndices ){
        std::ofstream myfile;
        myfile.open (fileName);

        for (int i = 0; i < temp_vertices.size(); i++){
            myfile << "v " << temp_vertices[i].x() << " " << temp_vertices[i].y() << " " << temp_vertices[i].z() << "\n";
        }

        for (int i = 0; i < vertexIndices.size(); i++){
            myfile << "f " << vertexIndices[i].x() << " " << vertexIndices[i].y() << " " << vertexIndices[i].z() << "\n";
        }
        myfile.close();
   }

   //OpenGL calls this method constantly to update the screen.
   virtual void drawGL() override {
      using namespace nanogui;

      //refer to the previous explanation of mShader.bind();
      mShader.bind();

      //this simple command updates the positions matrix. You need to do the same for color and indices matrices too
      //positions *= 100;
      mShader.uploadAttrib("vertexPosition_modelspace", positions);
      mShader.uploadAttrib("vertexNormal_modelspace", normals);
      mShader.uploadAttrib("color", colors);
      

      //This is a way to perform a simple rotation using a 4x4 rotation matrix represented by rmat
      //mvp stands for ModelViewProjection matrix
      Matrix4f mvp;
      mvp.setIdentity();
      // mvp.topLeftCorner<3, 3>() = Eigen::Matrix3f(Eigen::AngleAxisf(mRotation[0], Vector3f::UnitX()) *
      //                                             Eigen::AngleAxisf(mRotation[1], Vector3f::UnitY()) *
      //                                             Eigen::AngleAxisf(mRotation[2], Vector3f::UnitZ())) * 0.25f;

      // Matrix3f scaling_matrix;
      // scaling_matrix.setIdentity();

      //scaling_matrix *= 20;
      //mvp.topLeftCorner<3, 3>() = scaling_matrix * mvp.topLeftCorner<3, 3>();


      Matrix4f rotationMVP;
      rotationMVP.setIdentity();
      rotationMVP.topLeftCorner<3, 3>() = Eigen::Matrix3f(Eigen::AngleAxisf(mRotation[0], Vector3f::UnitX()) *
                                                Eigen::AngleAxisf(mRotation[1],  Vector3f::UnitY()) *
                                                Eigen::AngleAxisf(mRotation[2], Vector3f::UnitZ())) * 0.25f;



      Matrix4f scalingMVP;
      scalingMVP.setIdentity();
      scalingMVP.diagonal<0>()[0] = scaleValue;
      scalingMVP.diagonal<0>()[1] = scaleValue;
      scalingMVP.diagonal<0>()[2] = scaleValue;

      mvp = rotationMVP * scalingMVP;

      Matrix4f translationMVP;
      mvp.rightCols<1>()[0] = translationValues[0];
      mvp.rightCols<1>()[1] = translationValues[1];
      
      mShader.setUniform("MVP", mvp);

      // If enabled, does depth comparisons and update the depth buffer.
      // Avoid changing if you are unsure of what this means.
      glEnable(GL_DEPTH_TEST);

      /* Draw 12 triangles starting at index 0 of your indices matrix */
      /* Try changing the first input with GL_LINES, this will be useful in the assignment */
      /* Take a look at this link to better understand OpenGL primitives */
      /* https://www.khronos.org/opengl/wiki/Primitive */

      int mesh_positions_size = positions.size() / 3 - wireframe_positions.size() / 3;
      int wireframe_positions_size = positions.size() / 3;

      //12 triangles, each has three vertices
      if (meshRepresentVal == 0) {
         mShader.drawArray(GL_TRIANGLES, 0, mesh_positions_size);
         normals = flatNormals;
      }
      else if (meshRepresentVal == 1) {
         mShader.drawArray(GL_TRIANGLES, 0, mesh_positions_size);
         normals = smoothNormals;
      }
      else if (meshRepresentVal == 2) {
         mShader.drawArray(GL_LINES, mesh_positions_size, wireframe_positions_size);
      }
      else if (meshRepresentVal == 3) {
        normals = smoothNormals;
         mShader.drawArray(GL_TRIANGLES, 0, positions.size());
         mShader.drawArray(GL_LINES, mesh_positions_size, wireframe_positions_size);
      }

      //2 triangles, each has 3 lines, each line has 2 vertices
      //mShader.drawArray(GL_LINES, 12*3, 2*3*2);

      //mShader.drawIndexed(GL_TRIANGLES, 0, 12);
      //mShader.drawIndexed(GL_LINES, 12, 12);
      glDisable(GL_DEPTH_TEST);
   }

    MatrixXf flatNormals; 
    MatrixXf smoothNormals;
    MatrixXf wireframe_positions;
    MatrixXf mesh_positions;
    vector<nanogui::Vector3f> vSave;
    vector<nanogui::Vector3f> fSave;

   //Instantiation of the variables that can be acessed outside of this class to interact with the interface
   //Need to be updated if a interface element is interacting with something that is inside the scope of MyGLCanvas
   private:
   Vector3f translationValues = Vector3f(0,0,0);
   MatrixXf positions;
   MatrixXf normals;
   MatrixXf colors;
   nanogui::GLShader mShader;
   Eigen::Vector3f mRotation;
   int meshRepresentVal = 0;
   float scaleValue = 0.5;
};

class winged_edge_data_struct {
   
   public:
   winged_edge_data_struct(){

   }
   vector<Vertex> vertices;
   vector<Face> faces;
   vector<W_edge> edges;

   void decimateEdge(W_edge edge){
      
   }  

   template <class T>
   void removeObject(T object, vector<T> &vec){
      for (int i = 0; i < vec.size(); i++){
         if (vec[i] == object){
            vec.erase(vec.begin() + i);
            return;
         }
      }
      //cout << "Not found"<<endl;
   }

   template <class T>
   int getIndex(T object, vector<T> &vec){
      for (int i = 0; i < vec.size(); i++){
         if (vec[i] == object){
            return i;
         }
      }
      //cout << "Not found"<<endl;
      return -1;
   }

   void decimateMeshFromUI(MyGLCanvas *mCanvas, int dTimes, int dEdges){
      cout << "Decimate for "<< dTimes << endl;
      for (int i = 0; i < dTimes; i++){
         cout << "-----------------------------------------------------"<<endl;
         cout << "Decimating for iteration: "<< i+1 <<endl;
         cout << "Taking " << dEdges << " edges into consideration."<<endl;
         decimateMesh(mCanvas, dEdges);
      }
   }

   void decimateMesh(MyGLCanvas *mCanvas, int dEdges){
      
      MatrixXf new_vertice;
      MatrixXf edge_quadric;
      float edge_quadric_error;

      int selected_index;
      MatrixXf selected_edge_new_vertices;
      MatrixXf selected_edge_quadrics;
      float selected_edge_quadric_errors;
      float min_edge_error = 0.0;
      int edge_index = 0;
      vector<int> selected_edges;

      // pick dEdges number of random edges.
      while(selected_edges.size() != dEdges){
         edge_index = rand() % edges.size();
         std::vector<int>::iterator it = std::find(selected_edges.begin(), selected_edges.end(), 22);	

         if (it != selected_edges.end()){
            //std::cout << "Element Found" << std::endl;
         }else{
            //std::cout << "Element Not Found" << std::endl;
            selected_edges.push_back(edge_index);
         }
      }
      
      // find edge with minimum quadric error
      for (int i = 0; i < selected_edges.size(); i++){

         edge_index = selected_edges[i];
         
         edge_quadric = findEdgeQuadric(&edges[edge_index]);
         new_vertice = findOptimalContractionTarget(edge_quadric);
         edge_quadric_error = findEdgeQuadricError(edge_quadric, new_vertice);

         if(i == 0){
            min_edge_error = edge_quadric_error;
            selected_index = edge_index;
            selected_edge_quadrics = edge_quadric;
            selected_edge_new_vertices = new_vertice;
            selected_edge_quadric_errors = edge_quadric_error;
         }else if (edge_quadric_error < min_edge_error){
            min_edge_error = edge_quadric_error;
            selected_index = edge_index;
            selected_edge_quadrics = edge_quadric;
            selected_edge_new_vertices = new_vertice;
            selected_edge_quadric_errors = edge_quadric_error;
         }
      }
      selected_edges.clear();
      int idx = selected_index;

      cout << "Edge To Decimate: " << edges[idx] <<endl;
      //cout <<"Edge quadric: " << edge_quadric << endl;
      cout <<"Vertex to Decimate: " << *edges[idx].start << endl;
      cout <<"New Vertex position v: " << selected_edge_new_vertices.transpose() << endl;
      cout <<"Edge Quadric error: " << selected_edge_quadric_errors << endl;
      
      // decimate that edge
      edges[idx].start->decimated = true;
      edges[idx].left_next->left_prev->decimated = true;
      edges[idx].right_prev->left_next->decimated = true;
      edges[idx].right_next->decimated = true;
      edges[idx].right_next->right_prev->left_next->decimated = true;
      edges[idx].left_prev->decimated = true;
      edges[idx].left_prev->right_prev->left_next->decimated = true;
      edges[idx].right->decimated = true;
      edges[idx].left->decimated = true;

      // fix vertices
      Vertex *v = edges[idx].start;
      W_edge *e0 = v->edge;
      W_edge *edge_t = e0;
      vector<W_edge *> temp_edges_left;
      vector<W_edge *> temp_edges_right;
      do {
         if(edge_t->end == v){
            edge_t = edge_t->left_next;
         }else{
            edge_t = edge_t->right_next;
         }
         if(!edge_t->decimated){
            temp_edges_left.push_back(edge_t);
            temp_edges_right.push_back(edge_t->right_prev->left_next);
         }
         // cout << "left e: " << *edge_t << endl;
         // cout << "right e: " <<*edge_t->right_prev->left_next << endl;
         // cout << "edge start: " << *v << endl;
      }while (edge_t != e0);
      cout<< temp_edges_left.size() << endl;

      for(int i=0; i<temp_edges_left.size(); i++) {
         temp_edges_left[i]->start = edges[idx].end;
      }

      for(int i=0; i<temp_edges_right.size(); i++) {
         temp_edges_right[i]->end = edges[idx].end;
      }

      // fixing faces
      edges[idx].right_prev->left = edges[idx].right_next->right;
      edges[idx].left_next->left = edges[idx].left_prev->right;
      edges[idx].right_prev->right_prev->left_next->right = edges[idx].right_next->right;
      edges[idx].left_next->right_prev->left_next->right = edges[idx].left_prev->right;

      //wing face if they share the deleted edge
      if(edges[idx].right_next->right->edge == edges[idx].right_next->right_prev->left_next){
         edges[idx].right_next->right->edge = edges[idx].right_prev;
      }

      if(edges[idx].left_prev->right->edge == edges[idx].left_prev->right_prev->left_next){
         edges[idx].left_prev->right->edge = edges[idx].left_next;
      }

      // wing vertices if they share the deleted edge
      if(edges[idx].right_next->end->edge == edges[idx].right_next->right_prev->left_next){
         edges[idx].right_next->end->edge = edges[idx].right_next->right_prev->right_prev->left_next;
      }

      if(edges[idx].left_prev->start->edge == edges[idx].left_prev){
         edges[idx].left_prev->start->edge = edges[idx].left_prev->right_next;
      }
      
      // outer
      edges[idx].right_next->right_prev->right_prev->left_next->right_next = edges[idx].right_prev;
      edges[idx].left_prev->right_next->right_prev->left_next->right_prev = edges[idx].left_next;

      edges[idx].right_prev->right_prev->left_next->right_prev = edges[idx].right_next->right_prev;
      edges[idx].left_next->right_prev->left_next->right_next = edges[idx].left_prev->right_next;

      edges[idx].right_prev->right_prev->left_next->right_prev = edges[idx].right_next->right_next;
      edges[idx].left_next->right_prev->left_next->right_next = edges[idx].left_prev->right_prev;

      edges[idx].right_next->right_next->right_prev->left_next->right_prev = edges[idx].right_prev;
      edges[idx].left_prev->right_prev->right_prev->left_next->right_next = edges[idx].left_next;

      // inner
      edges[idx].right_next->right_prev->left_next = edges[idx].right_prev;
      edges[idx].left_prev->right_next->left_prev = edges[idx].left_next;

      edges[idx].right_prev->left_prev = edges[idx].right_next->right_prev;
      edges[idx].left_next->left_next = edges[idx].left_prev->right_next;

      edges[idx].right_prev->left_next = edges[idx].right_next->right_next;
      edges[idx].left_next->left_prev = edges[idx].left_prev->right_prev;

      edges[idx].right_next->right_next->left_prev = edges[idx].right_prev;
      edges[idx].left_prev->right_prev->left_next = edges[idx].left_next;

      // cout << "removing edge: " << edges[idx] << endl;
      // cout << "removing vertex: " << *edges[idx].start << endl;

      // update the edge's end to new vertex
      edges[idx].end->quadric_matrix = edge_quadric;
      edges[idx].end->is_new = true;
      edges[idx].end->x = selected_edge_new_vertices(0,0);
      edges[idx].end->y = selected_edge_new_vertices(1,0);
      edges[idx].end->z = selected_edge_new_vertices(2,0);

      cout << "-----------------------------------------------------"<<endl;

      updateMesh(mCanvas);
   }

   float findEdgeQuadricError(MatrixXf quad_edge, MatrixXf v){
      MatrixXf error = v.transpose() * quad_edge * v;
      return error(0,0);
   }

   MatrixXf findOptimalContractionTarget(MatrixXf quad_edge){
      
      MatrixXf t = MatrixXf(1,4);
      t << 0, 0, 0, 1;

      quad_edge.row(3) << t.row(0);

      MatrixXf v = quad_edge.inverse() * t.transpose();
      return v;
   }

   MatrixXf findEdgeQuadric(W_edge *edge){

      MatrixXf quad_edge_error = MatrixXf(4,4);
      quad_edge_error = findVertexQuadric(edge->start) + findVertexQuadric(edge->end);
      return quad_edge_error;      
   }

   MatrixXf findVertexQuadric(Vertex *v){
      MatrixXf quad_vertex_error;

      if (!(v->is_new)){
         // finding all the faces that share the vertice
         W_edge *e0 = v->edge;
         W_edge *edge = e0;
         quad_vertex_error = MatrixXf::Zero(4,4);
         do{
            if(edge->end == v){
               quad_vertex_error += findFaceQuadricError(edge->left);
               edge = edge->left_next;
            }else{
               quad_vertex_error += findFaceQuadricError(edge->right);
               edge = edge->right_next;
            }
         } while(edge != e0);
      }else{
        quad_vertex_error = v->quadric_matrix;
      }
      
      return quad_vertex_error;
   }

   MatrixXf findFaceQuadricError(Face *f){
      // Finding Plane equation
      MatrixXf vert = MatrixXf(3,3);

      vert.col(0) << f->edge->end->x, f->edge->end->y, f->edge->end->z; 

      vert.col(1) << f->edge->left_next->end->x, f->edge->left_next->end->y, f->edge->left_next->end->z ;

      vert.col(2) << f->edge->start->x, f->edge->start->y, f->edge->start->z ;

      Vector3f s1 = vert.col(1) - vert.col(0);
      Vector3f s2 = vert.col(2) - vert.col(0);

      Vector3f face_normal;
      face_normal.x() = (s1.y()*s2.z()) - (s1.z()*s2.y());
      face_normal.y() = (s1.z()*s2.x()) - (s1.x()*s2.z());
      face_normal.z() = (s1.x()*s2.y()) - (s1.y()*s2.x());

      float a = face_normal.x();
      float b = face_normal.y();
      float c = face_normal.z();
      float d = -(a*vert.col(0)[0] + b*vert.col(0)[1] + c*vert.col(0)[2]);

      MatrixXf p = MatrixXf(1,4);
      p.col(0) << a, b, c, d;

      return p.transpose() * p;
   }

   void saveMeshtoFile(string fileName){
      std::ofstream myfile;
      myfile.open (fileName);

      vector<Vertex> temp_vertices;
      vector<Vector3f> vertexIndices;

      for (int i = 0; i < vertices.size(); i++){
         if(!vertices[i].decimated){
            bool save = true;
            int index = getIndex(vertices[i], temp_vertices);
            if(index == -1){
               temp_vertices.push_back(vertices[i]);
               save = true;
            }
            cout<<temp_vertices[i]<<endl;
         }
      }

      for (int i = 0; i < faces.size(); i++){
         W_edge *e0 = faces[i].edge;
         W_edge *edge = e0;
         Vertex *vertex;
         Vector3f vert_ind;
         vector<Vertex> face_vertices;

         do {
            if(*edge->right == faces[i]) {
               edge = edge->right_prev;
               vertex = edge->end;
            }else{
               edge = edge->left_prev;
               vertex = edge->start;
            }
            if(!faces[i].decimated){
               face_vertices.push_back(*vertex);
            }
         }while(edge != e0);

         for (int j = 0; j < face_vertices.size(); j++){
            vert_ind[j] = getIndex(face_vertices[j], temp_vertices) + 1;
         }

         vertexIndices.push_back(vert_ind);
         face_vertices.clear();
      }

      for (int i = 0; i < temp_vertices.size(); i++){
         myfile << "v " << temp_vertices[i].x << " " << temp_vertices[i].y << " " << temp_vertices[i].z << "\n";
      }

      for (int i = 0; i < vertexIndices.size(); i++){
         myfile << "f " << vertexIndices[i].x() << " " << vertexIndices[i].y() << " " << vertexIndices[i].z() << "\n";
      }
      myfile.close();
   }

    MatrixXf generateSmoothNormals(){
        MatrixXf sNormals = MatrixXf(3, faces.size() * 3);

         cout << "Sie: " << vertices.size()<< endl;
        for (int i = 0; i < vertices.size(); i++){
            cout << i << ": " << *vertices[i].edge << endl;
            W_edge *e0 = vertices[i].edge;
            W_edge *edge = e0;
            Vector3f s_normal = Vector3f(0, 0, 0);
            int count = 0;
            do {
                  cout << "While" << endl;
                if(edge->end == &vertices[i]){
                   cout << "END" << endl;
                    count++;
                    Face *f = edge->left;
                    MatrixXf vert = MatrixXf(3,3);

                    vert.col(0) << f->edge->end->x, f->edge->end->y, f->edge->end->z; 

                    vert.col(1) << f->edge->left_next->end->x, f->edge->left_next->end->y, f->edge->left_next->end->z ;

                    vert.col(2) << f->edge->start->x, f->edge->start->y, f->edge->start->z ;

                    Vector3f s1 = vert.col(1) - vert.col(0);
                     Vector3f s2 = vert.col(2) - vert.col(0);

                     Vector3f face_normal;
                     face_normal.x() = (s1.y()*s2.z()) - (s1.z()*s2.y());
                     face_normal.y() = (s1.z()*s2.x()) - (s1.x()*s2.z());
                     face_normal.z() = (s1.x()*s2.y()) - (s1.y()*s2.x());
            
                    face_normal = face_normal.normalized();
                    s_normal = s_normal + face_normal;
        
                    edge = edge->left_next;    
                    
                }else{

                   cout << "START" << endl;
                    count++;
                    Face *f = edge->right;
                    MatrixXf vert = MatrixXf(3,3);

                    vert.col(0) << f->edge->end->x, f->edge->end->y, f->edge->end->z; 

                    vert.col(1) << f->edge->left_next->end->x, f->edge->left_next->end->y, f->edge->left_next->end->z ;

                    vert.col(2) << f->edge->start->x, f->edge->start->y, f->edge->start->z ;

                    Vector3f s1 = vert.col(1) - vert.col(0);
                     Vector3f s2 = vert.col(2) - vert.col(0);

                     Vector3f face_normal;
                     face_normal.x() = (s1.y()*s2.z()) - (s1.z()*s2.y());
                     face_normal.y() = (s1.z()*s2.x()) - (s1.x()*s2.z());
                     face_normal.z() = (s1.x()*s2.y()) - (s1.y()*s2.x());

                     face_normal = face_normal.normalized();
                    s_normal = s_normal + face_normal;

                    edge = edge->right_next;
                }
            } while(edge != e0);

            s_normal = s_normal / count;

            sNormals.col(i) << s_normal.x(), s_normal.y(), s_normal.z();
        }   
    
        return sNormals;
    }

    MatrixXf generateFlatNormals(){
        MatrixXf fNormals = MatrixXf(3, faces.size() * 3);

        for (int i = 0; i < faces.size(); i++){
            Face *f = &faces[i];
            MatrixXf vert = MatrixXf(3,3);

            vert.col(0) << f->edge->end->x, f->edge->end->y, f->edge->end->z; 

            vert.col(1) << f->edge->left_next->end->x, f->edge->left_next->end->y, f->edge->left_next->end->z ;

            vert.col(2) << f->edge->start->x, f->edge->start->y, f->edge->start->z ;

            Vector3f s1 = vert.col(1) - vert.col(0);
            Vector3f s2 = vert.col(2) - vert.col(0);

            Vector3f normal;
            normal.x() = (s1.y()*s2.z()) - (s1.z()*s2.y());
            normal.y() = (s1.z()*s2.x()) - (s1.x()*s2.z());
            normal.z() = (s1.x()*s2.y()) - (s1.y()*s2.x());

            for (int j = 0; j < 3; j++){
                fNormals.col(i * 3 + j) = normal;
            }
        }
        return fNormals;
    }

      void updateMesh(MyGLCanvas *mCanvas){

         MatrixXf newPositions;
         MatrixXf wireframe_positions;

         positions_from_wingedEdge(newPositions, wireframe_positions);
            //newPositionss *= 20;

         mCanvas->updateMeshPositions(newPositions, wireframe_positions);
         mCanvas->updateMeshColors(newPositions, wireframe_positions);

         MatrixXf flatNormals = generateFlatNormals();
         //MatrixXf smoothNormals = generateSmoothNormals();
         mCanvas->flatNormals = flatNormals;
         //mCanvas->smoothNormals = smoothNormals;
      }

      void obj_to_winged_edge(MyGLCanvas *mCanvas, string file_name, vector<unsigned int> &vertexIndices, vector<Vector3f> &temp_vertices) {

         temp_vertices.clear();
         vertexIndices.clear();

         mCanvas->vSave.clear();
         mCanvas->fSave.clear();

         edges.clear();
         faces.clear();
         vertices.clear();

         // Record start time
         auto start = std::chrono::high_resolution_clock::now();

         const char * fileName = file_name.c_str();

         FILE * file = fopen(fileName, "r");
         if (file == NULL) 
         {
            printf("Impossible to open file !\n");
         }

         while (true)
         {
            char lineHeader[128];

            int res = fscanf(file, "%s", lineHeader);
            if (res == EOF)
               break; // EOF = End Of File. Quit the loop.

            if (strcmp(lineHeader, "v") == 0) 
            {
               Vector3f vertex;
               fscanf(file, "%f %f %f \n", &vertex[0], &vertex[1], &vertex[2]);
               temp_vertices.push_back(vertex);
               mCanvas->vSave.push_back(vertex);
            }
            else if (strcmp(lineHeader, "f") == 0) 
            {
               unsigned int vertexIndex[3];
               fscanf(file, "%i %i %i \n", &vertexIndex[0], &vertexIndex[1], &vertexIndex[2]);
               vertexIndices.push_back(vertexIndex[0] - 1);
               vertexIndices.push_back(vertexIndex[1] - 1);
               vertexIndices.push_back(vertexIndex[2] - 1);

               mCanvas->fSave.push_back(Vector3f(vertexIndex[0], vertexIndex[1], vertexIndex[2]));
            }

         }

         fclose(file);

         int numberofVertices = temp_vertices.size();
         int numberofFaces = vertexIndices.size() / 3;
         int numberofEdges = numberofVertices + numberofFaces - 2;
         cout<<"Number of edges * 2   =    "<<(numberofEdges*2)<<endl;


        Vector3f max = Vector3f(-1, -1, -1);
        Vector3f min = Vector3f(1, 1, 1);

        for (int i = 0; i < temp_vertices.size(); i++){
            for (int j = 0; j < 3; j++){
                if (max[j] < temp_vertices[i][j]){
                    max[j] = temp_vertices[i][j];
                }
                if (min[j] > temp_vertices[i][j]){
                    min[j] = temp_vertices[i][j];
                }
            }
        }

        for (int i = 0; i < temp_vertices.size(); i++){
            // normalize
            for (int k = 0; k < 3; k++){
                temp_vertices[i][k] = 2 * (temp_vertices[i][k] - min[k]) / (max - min)[k] - 1;
            }
        }

        for(int i=0; i<numberofVertices; i++)
         {
            Vertex v;
            v.x = temp_vertices[i].x();
            v.y = temp_vertices[i].y();
            v.z = temp_vertices[i].z();
            v.edge = NULL;
            v.is_new = false;
            v.decimated = false;
            vertices.push_back(v);
         }

        for (int i = 0; i<numberofFaces; i++) 
         {
            Face f;
            f.edge = NULL;
            f.decimated = false;
            faces.push_back(f);
         }

        for (int i = 0; i < numberofEdges*2; i++)
         {

            W_edge e;
            e.start = NULL;
            e.end = NULL;
            e.left = NULL;
            e.right = NULL;
            e.left_next = NULL;
            e.left_prev = NULL;
            e.right_next = NULL;
            e.right_prev = NULL;
            e.decimated = false;
            edges.push_back(e);
         }

        for (int i = 0; i < numberofEdges*2; i++)
         {

            int j = i + 1;

            if (j % 3 == 1)
            {
               edges[i].start = &vertices[vertexIndices[i]];
               if ((vertices[vertexIndices[i]].edge) == NULL) 
               {
                  vertices[vertexIndices[i]].edge = &edges[i];
               }

               edges[i].end = &vertices[vertexIndices[j]];
      
               edges[i].left_next = &edges[j];
               edges[i].left_prev = &edges[j+1];

            }
            else if(j % 3 == 2)
            {
                edges[i].start = &vertices[vertexIndices[i]];
               if ((vertices[vertexIndices[i]].edge) == NULL) 
               {
                  vertices[vertexIndices[i]].edge = &edges[i];
               }

               edges[i].end = &vertices[vertexIndices[j]];
          

               edges[i].left_next = &edges[j];
               edges[i].left_prev = &edges[i-1];
            }
            else 
            {
               edges[i].end = &vertices[vertexIndices[i - 2]];
       

               edges[i].start = &vertices[vertexIndices[i]];
               if ((vertices[vertexIndices[i]].edge) == NULL)
               {
                  vertices[vertexIndices[i]].edge = &edges[i];
               }

               if ((faces[(j / 3) - 1].edge) == NULL)
               {
                  faces[(j / 3) - 1].edge = &edges[i];
               }

               edges[i].left = &faces[(j / 3) - 1];
               edges[i - 1].left = &faces[(j / 3) - 1];
               edges[i - 2].left = &faces[(j / 3) - 1];

               edges[i].left_next = &edges[i-2];
               edges[i].left_prev = &edges[i-1];

            }
         }

         for (int i = 0; i < numberofEdges*2; i++) 
         {
            for (int j = 0; j < numberofEdges*2; j++) 
            {
              
                  if (edges[j].start == edges[i].end && edges[j].end == edges[i].start)
                  {
                     edges[i].right = edges[j].left;
                     edges[i].right_next = edges[j].left_next;
                     edges[i].right_prev = edges[j].left_prev;
                     break;
                  }
               
            }
         }

         // Record end time
         auto finish = std::chrono::high_resolution_clock::now();
         std::chrono::duration<double> elapsed = finish - start;
         //std::cout << "Elapsed time: " << elapsed.count() << " s\n";
         //updateMesh(mCanvas);
      }

      void positions_from_wingedEdge(MatrixXf &poses, MatrixXf &wireframe_poses)
      {  
         poses = MatrixXf (3, faces.size()*3);
         wireframe_poses = MatrixXf(3, faces.size() * 6);

         int j = 0;
         int k = 0;
         for (int i = 0; i < faces.size(); i++ )
         {  
            Face *f = &faces[i];
            if(!f->decimated){
               poses.col(j) << f->edge->end->x, f->edge->end->y, f->edge->end->z ;
               j++; 

               poses.col(j) << f->edge->left_next->end->x, f->edge->left_next->end->y, f->edge->left_next->end->z ;
               j++;

               poses.col(j) << f->edge->start->x, f->edge->start->y, f->edge->start->z ;
               j++;

               wireframe_poses.col(k) << poses.col(j - 3);
               wireframe_poses.col(k + 1) << poses.col(j - 2);
               wireframe_poses.col(k + 2) << poses.col(j - 2);
               wireframe_poses.col(k + 3) << poses.col(j - 1);
               wireframe_poses.col(k + 4) << poses.col(j - 1);
               wireframe_poses.col(k + 5) << poses.col(j - 3);
               k += 6;
            }
         } 
      }
};


class ExampleApplication : public nanogui::Screen {
   public:
   ExampleApplication() : nanogui::Screen(Eigen::Vector2i(900, 650), "NanoGUI Cube and Menus", false) {
      using namespace nanogui;

      //OpenGL canvas demonstration

      //First, we need to create a window context in which we will render both the interface and OpenGL canvas
      Window *window = new Window(this, "GLCanvas Demo");
      window->setPosition(Vector2i(15, 15));
      window->setLayout(new GroupLayout());

      //OpenGL canvas initialization, we can control the background color and also its size
      mCanvas = new MyGLCanvas(window);
      mCanvas->setBackgroundColor({ 100, 100, 100, 255 });
      mCanvas->setSize({ 400, 400 });

      winged = new winged_edge_data_struct();

      //This is how we add widgets, in this case, they are connected to the same window as the OpenGL canvas
      Widget *tools = new Widget(window);
      tools->setLayout(new BoxLayout(Orientation::Horizontal,
         Alignment::Middle, 0, 5));

      //then we start adding elements one by one as shown below
      Button *b0 = new Button(tools, "Random Color");
      b0->setCallback([this]() { mCanvas->setBackgroundColor(Vector4i(rand() % 256, rand() % 256, rand() % 256, 255)); });

      Button *b1 = new Button(tools, "Random Rotation");
      b1->setCallback([this]() { mCanvas->updateRotationValues(nanogui::Vector3f((rand() % 100) / 100.0f, (rand() % 100) / 100.0f, (rand() % 100) / 100.0f)); });

      Button *b2 = new Button(tools, "Quit");
      b2->setCallback([this]() { exit(0); });

      //widgets demonstration
      nanogui::GLShader mShader;

      //Then, we can create another window and insert other widgets into it
      Window *anotherWindow = new Window(this, "Basic widgets");
      anotherWindow->setPosition(Vector2i(500, 15));
      anotherWindow->setLayout(new GroupLayout());

      // GUI for Decimation
      new Label(anotherWindow, "Mesh Decimation", "sans-bold");
      
      Widget *panelDecimation1 = new Widget(anotherWindow);
      panelDecimation1->setLayout(new BoxLayout(Orientation::Horizontal,
      Alignment::Middle, 0, 0));
      new Label(panelDecimation1, "Times to decimate", "sans-bold");
      //Demonstration of rotation along one axis of the mesh using a simple slider, you can have three of these, one for each dimension
      auto textBoxDecimateTimes = new IntBox<int>(panelDecimation1);
      textBoxDecimateTimes->setValue(1);
      textBoxDecimateTimes->setEditable(true);
         

      Widget *panelDecimation2 = new Widget(anotherWindow);
      panelDecimation2->setLayout(new BoxLayout(Orientation::Horizontal,
      Alignment::Middle, 0, 0));
      new Label(panelDecimation2, "Edges to consider", "sans-bold");
      auto textBoxNoOfEdges = new IntBox<int>(panelDecimation2);
      textBoxNoOfEdges->setValue(4);
      textBoxNoOfEdges->setEditable(true);
      // textBoxDecimateTimes->setCallback({
      //    cout << textBoxDecimateTimes->value()<< endl;
      // });

      Button *buttonDecimate = new Button(anotherWindow, "Decimate");
      buttonDecimate->setCallback([&, buttonDecimate, textBoxDecimateTimes, textBoxNoOfEdges] {
         
         int dTimes = textBoxDecimateTimes->value();
         int dEdges = textBoxNoOfEdges->value();    

         winged->decimateMeshFromUI(mCanvas, dTimes, dEdges);
      });

      // Demonstrates how a button called "New Mesh" can update the positions matrix.
      // This is just a demonstration, you actually need to bind mesh updates with the open file interface
      Button *button = new Button(anotherWindow, "New Mesh");

      button->setTooltip("Demonstrates how a button can update the positions matrix.");

      //this is how we write captions on the window, if you do not want to write inside a button 
      new Label(window, "Rotation on the first axis", "sans-bold");

      Widget *panelRotX = new Widget(anotherWindow);
      panelRotX->setLayout(new BoxLayout(Orientation::Horizontal,
      Alignment::Middle, 0, 0));

      new Label(panelRotX, "X Rotation", "sans-bold");
      //Demonstration of rotation along one axis of the mesh using a simple slider, you can have three of these, one for each dimension
      Slider *rotSliderX = new Slider(panelRotX);
      rotSliderX->setValue(0.5f);
      rotSliderX->setFixedWidth(150);
      rotSliderX->setCallback([&](float value) {
         float radians = (value - 0.5f) * 2 * 2 * M_PI;
         mCanvas->updateRotationValues(nanogui::Vector3f(radians, 0.0f, 0.0f));
      });


      Widget *panelRotY = new Widget(anotherWindow);
      panelRotY->setLayout(new BoxLayout(Orientation::Horizontal,
      Alignment::Middle, 0, 0));

      new Label(panelRotY, "Y Rotation", "sans-bold");
      //Demonstration of rotation along one axis of the mesh using a simple slider, you can have three of these, one for each dimension
      Slider *rotSliderY = new Slider(panelRotY);
      rotSliderY->setValue(0.5f);
      rotSliderY->setFixedWidth(150);
      rotSliderY->setCallback([&](float value) {
         float radians = (value - 0.5f) * 2 * 2 * M_PI;
         mCanvas->updateRotationValues(nanogui::Vector3f(0.0f, radians, 0.0f));
      });


      Widget *panelRotZ = new Widget(anotherWindow);
      panelRotZ->setLayout(new BoxLayout(Orientation::Horizontal,
      Alignment::Middle, 0, 0));

      new Label(panelRotZ, "Z Rotation", "sans-bold");
      //Demonstration of rotation along one axis of the mesh using a simple slider, you can have three of these, one for each dimension
      Slider *rotSliderZ = new Slider(panelRotZ);
      rotSliderZ->setValue(0.5f);
      rotSliderZ->setFixedWidth(150);
      rotSliderZ->setCallback([&](float value) {
         float radians = (value - 0.5f) * 2 * 2 * M_PI;
         mCanvas->updateRotationValues(nanogui::Vector3f(0.0f, 0.0f, radians));
      });

      Widget *panelScaling = new Widget(anotherWindow);
      panelScaling->setLayout(new BoxLayout(Orientation::Horizontal,
      Alignment::Middle, 0, 0));

      new Label(panelScaling, "Scale", "sans-bold");
      //Demonstration of rotation along one axis of the mesh using a simple slider, you can have three of these, one for each dimension
      Slider *scaleSlider = new Slider(panelScaling);
      scaleSlider->setValue(0.5f);
      scaleSlider->setFixedWidth(150);
      scaleSlider->setCallback([&](float value) {
         float scale = value * 2;
         mCanvas->updateScaleValues(scale);
      });

      Widget *panelTraslateX = new Widget(anotherWindow);
      panelTraslateX->setLayout(new BoxLayout(Orientation::Horizontal,
      Alignment::Middle, 0, 0));

      new Label(panelTraslateX, "translationX", "sans-bold");
      //Demonstration of rotation along one axis of the mesh using a simple slider, you can have three of these, one for each dimension
      Slider *transXSlider = new Slider(panelTraslateX);
      transXSlider->setValue(0.5f);
      transXSlider->setFixedWidth(150);
      transXSlider->setCallback([&](float value) {
         float translation = (value - 0.5f) * 2;
         mCanvas->updateTranslationValues(translation, 0);
      });

      Widget *panelTraslateY = new Widget(anotherWindow);
      panelTraslateY->setLayout(new BoxLayout(Orientation::Horizontal,
      Alignment::Middle, 0, 0));

      new Label(panelTraslateY, "translationY", "sans-bold");
      //Demonstration of rotation along one axis of the mesh using a simple slider, you can have three of these, one for each dimension
      Slider *transYSlider = new Slider(panelTraslateY);
      transYSlider->setValue(0.5f);
      transYSlider->setFixedWidth(150);
      transYSlider->setCallback([&](float value) {
         float translation = (value- 0.5f) * 2;
         mCanvas->updateTranslationValues(translation, 1);
      });

      //Message dialog demonstration, it should be pretty straightforward
      new Label(anotherWindow, "Message dialog", "sans-bold");
      tools = new Widget(anotherWindow);

      tools->setLayout(new BoxLayout(Orientation::Horizontal,
      Alignment::Middle, 0, 6));

      Button *b = new Button(tools, "Info");

      b->setCallback([&] {
         auto dlg = new MessageDialog(this, MessageDialog::Type::Information, "Title", "This is an information message");
         dlg->setCallback([](int result) { cout << "Dialog result: " << result << endl; });
      });

      b = new Button(tools, "Warn");

      b->setCallback([&] {
         auto dlg = new MessageDialog(this, MessageDialog::Type::Warning, "Title", "This is a warning message");
         dlg->setCallback([](int result) { cout << "Dialog result: " << result << endl; });
      });

      b = new Button(tools, "Ask");

      b->setCallback([&] {
         auto dlg = new MessageDialog(this, MessageDialog::Type::Warning, "Title", "This is a question message", "Yes", "No", true);
         dlg->setCallback([](int result) { cout << "Dialog result: " << result << endl; });
      });

      //Here is how you can get the string that represents file paths both for opening and for saving.
      //you need to implement the rest of the parser logic.
      new Label(anotherWindow, "File dialog", "sans-bold");
      tools = new Widget(anotherWindow);
      tools->setLayout(new BoxLayout(Orientation::Horizontal,
      Alignment::Middle, 0, 6));

      b = new Button(tools, "Open");
      b->setCallback([&] {
         std::string file_name = file_dialog({ {"obj", "Wavefront OBJ"} }, false);
         
         vector<unsigned int> vertexIndices;
         vector<Vector3f> temp_vertices;
         winged->obj_to_winged_edge(mCanvas, file_name, vertexIndices, temp_vertices);
         winged->updateMesh(mCanvas);
         // mCanvas->renderNewMesh(temp_vertices, vertexIndices);
      });

      b = new Button(tools, "Save");
      b->setCallback([&] {
        std::string file_name = file_dialog({ {"obj", "Wavefront OBJ"} }, true);
        //mCanvas->saveMeshtoObj(file_name, mCanvas->vSave, mCanvas->fSave);
        winged->saveMeshtoFile(file_name);
        cout << "File dialog result: " << file_name << endl;
      });

      //This is how to implement a combo box, which is important in A1
      new Label(anotherWindow, "Combo box", "sans-bold");
      ComboBox *combo = new ComboBox(anotherWindow, { "Flat Shaded", "Smooth Shaded", "Wireframe", "Shaded with Mesh Edges" });
      combo->setCallback([&](int value) {
         cout << "Combo box selected: " << value << endl;
         mCanvas->updateMeshRepresentVal(value);
      });

      new Label(anotherWindow, "Check box", "sans-bold");
      CheckBox *cb = new CheckBox(anotherWindow, "Flag 1",
      [](bool state) { cout << "Check box 1 state: " << state << endl; });
      
      cb->setChecked(true);
      cb = new CheckBox(anotherWindow, "Flag 2",
      [](bool state) { cout << "Check box 2 state: " << state << endl; });

      new Label(anotherWindow, "Progress bar", "sans-bold");
      mProgress = new ProgressBar(anotherWindow);

      new Label(anotherWindow, "Slider and text box", "sans-bold");

      Widget *panel = new Widget(anotherWindow);
      panel->setLayout(new BoxLayout(Orientation::Horizontal,
                        Alignment::Middle, 0, 20));

      //Fancy slider that has a callback function to update another interface element
      Slider *slider = new Slider(panel);
      slider->setValue(0.5f);
      slider->setFixedWidth(80);
      TextBox *textBox = new TextBox(panel);
      textBox->setFixedSize(Vector2i(60, 25));
      textBox->setValue("50");
      textBox->setUnits("%");
      slider->setCallback([textBox](float value) {
       textBox->setValue(std::to_string((int)(value * 100)));
      });
      slider->setFinalCallback([&](float value) {
         cout << "Final slider value: " << (int)(value * 100) << endl;
      });
      textBox->setFixedSize(Vector2i(60, 25));
      textBox->setFontSize(20);
      textBox->setAlignment(TextBox::Alignment::Right);

      //Method to assemble the interface defined before it is called
      performLayout();
   }

   //This is how you capture mouse events in the screen. If you want to implement the arcball instead of using
   //sliders, then you need to map the right click drag motions to suitable rotation matrices
   virtual bool mouseMotionEvent(const Eigen::Vector2i &p, const Vector2i &rel, int button, int modifiers) override {
      if (button == GLFW_MOUSE_BUTTON_3) {
         //Get right click drag mouse event, print x and y coordinates only if right button pressed
         cout << p.x() << " " << p.y() << "\n";
         return true;
      }
      return false;
   }

   virtual void drawContents() override {
      // ... put your rotation code here if you use dragging the mouse, updating either your model points, the mvp matrix or the V matrix, depending on the approach used
   }

   virtual void draw(NVGcontext *ctx) {
      /* Animate the scrollbar */
      mProgress->setValue(std::fmod((float)glfwGetTime() / 10, 1.0f));

      /* Draw the user interface */
      Screen::draw(ctx);
   }

   private:
   nanogui::ProgressBar *mProgress;
   MyGLCanvas *mCanvas;
   winged_edge_data_struct *winged;
};

int main(int /* argc */, char ** /* argv */) {
   try {
      nanogui::init();

      /* scoped variables */ {
         nanogui::ref<ExampleApplication> app = new ExampleApplication();
         app->drawAll();
         app->setVisible(true);
         nanogui::mainloop();
      }

      nanogui::shutdown();
   }
   catch (const std::runtime_error &e) {
      std::string error_msg = std::string("Caught a fatal error: ") + std::string(e.what());
      #if defined(_WIN32)
      MessageBoxA(nullptr, error_msg.c_str(), NULL, MB_ICONERROR | MB_OK);
      #else
      std::cerr << error_msg << endl;
      #endif
      return -1;
   }

return 0;
}
