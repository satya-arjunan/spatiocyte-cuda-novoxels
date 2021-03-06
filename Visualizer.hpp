//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of the Spatiocyte package
//
//        Copyright (C) 2006-2009 Keio University
//        Copyright (C) 2010-2013 RIKEN
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// Spatiocyte is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// Spatiocyte is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with Spatiocyte -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Satya Arjunan <satya.arjunan@gmail.com>
//


#ifndef __Visualizer_hpp
#define __Visualizer_hpp

#define SPHERE          1
#define BOX             2
#define GRID            3

//Lattice type:
#define HCP_LATTICE   0
#define CUBIC_LATTICE 1

using namespace std;

struct Color
{
  float r;
  float g;
  float b;
};

struct Point 
{
  double  x;
  double  y;
  double  z;
};

class GLScene;

class ControlBox : public Gtk::ScrolledWindow
{
public:
  ControlBox(GLScene&, Gtk::Table&);
  virtual ~ControlBox();
  void resizeScreen(unsigned, unsigned);
  void set_frame_cnt(int);
  void setTime(double);
  void setXangle(double);
  void setYangle(double);
  void setZangle(double);
  void play();
  void pause();
  void step();
  void play_or_pause();
protected:
  bool isChanging;
  bool on_background_clicked(GdkEventButton*);
  bool on_checkbutton_clicked(GdkEventButton*, unsigned int);
  unsigned int theSpeciesSize;
  void onResetBound();
  void onResetRotation();
  void on_3DMolecule_toggled();
  void on_InvertBound_toggled();
  void on_checkbutton_toggled(unsigned int id);
  void on_record_toggled();
  void on_resetTime_clicked();
  void on_showSurface_toggled();
  void on_showTime_toggled();
  void update_background_color(Gtk::ColorSelection*);
  void update_species_color(unsigned int, Gtk::ColorSelection*);
  void xLowBoundChanged();
  void xRotateChanged();
  void xUpBoundChanged();
  void yLowBoundChanged();
  void yRotateChanged();
  void yUpBoundChanged();
  void zLowBoundChanged();
  void zRotateChanged();
  void zUpBoundChanged();
  void progress_changed();
  void progress_adjust();
protected:
  GLScene& m_area_;
  Gtk::CheckButton** theButtonList;
  Gtk::Label** theLabelList;
  Gtk::Table m_table;
  Gtk::Table& m_area_table_;
private:
  Gdk::Color theBgColor;
  Glib::RefPtr<Gtk::SizeGroup> m_sizeGroup;
  Gtk::Adjustment theDepthAdj;
  Gtk::Adjustment theXAdj;
  Gtk::Adjustment theXLowBoundAdj;
  Gtk::Adjustment theXUpBoundAdj;
  Gtk::Adjustment theYAdj;
  Gtk::Adjustment theYLowBoundAdj;
  Gtk::Adjustment theYUpBoundAdj;
  Gtk::Adjustment theZAdj;
  Gtk::Adjustment theZLowBoundAdj;
  Gtk::Adjustment theZUpBoundAdj;
  Gtk::Button theResetTimeButton;
  Gtk::Button theResetBoundButton;
  Gtk::Button theResetDepthButton;
  Gtk::Button theResetRotButton;
  Gtk::CheckButton theCheck3DMolecule;
  Gtk::CheckButton theCheckInvertBound;
  Gtk::CheckButton theCheckShowSurface;
  Gtk::CheckButton theCheckShowTime;
  Gtk::Entry frame_cnt_;
  Gtk::Entry m_time;
  Gtk::Entry m_width;
  Gtk::Entry m_height;
  Gtk::Frame theFrameBoundAdj;
  Gtk::Frame theFrameLatticeAdj;
  Gtk::Frame theFrameRotAdj;
  Gtk::Frame theFrameScreen;
  Gtk::HBox m_rightBox;
  Gtk::HBox m_timeBox;
  Gtk::HBox the3DMoleculeBox;
  Gtk::HBox theBoxBoundFixReset;
  Gtk::HBox theBoxRotFixReset;
  Gtk::HBox theDepthBox;
  Gtk::HBox theHeightBox;
  Gtk::HBox theWidthBox;
  Gtk::HBox theXBox;
  Gtk::HBox theXLowBoundBox;
  Gtk::HBox theXUpBoundBox;
  Gtk::HBox theYBox;
  Gtk::HBox theYLowBoundBox;
  Gtk::HBox theYUpBoundBox;
  Gtk::HBox theZBox;
  Gtk::HBox theZLowBoundBox;
  Gtk::HBox theZUpBoundBox;
  Gtk::HScale theDepthScale;
  Gtk::HScale theXLowBoundScale;
  Gtk::HScale theXScale;
  Gtk::HScale theXUpBoundScale;
  Gtk::HScale theYLowBoundScale;
  Gtk::HScale theYScale;
  Gtk::HScale theYUpBoundScale;
  Gtk::HScale theZLowBoundScale;
  Gtk::HScale theZScale;
  Gtk::HScale theZUpBoundScale;
  Gtk::Label m_bgColor;
  Gtk::Label frame_cnt_label_;
  Gtk::Label m_timeLabel;
  Gtk::Label theDepthLabel;
  Gtk::Label theHeightLabel;
  Gtk::Label theWidthLabel;
  Gtk::Label theXLabel;
  Gtk::Label theXLowBoundLabel;
  Gtk::Label theXUpBoundLabel;
  Gtk::Label theYLabel;
  Gtk::Label theYLowBoundLabel;
  Gtk::Label theYUpBoundLabel;
  Gtk::Label theZLabel;
  Gtk::Label theZLowBoundLabel;
  Gtk::Label theZUpBoundLabel;
  Gtk::SpinButton theDepthSpin;
  Gtk::SpinButton theXLowBoundSpin;
  Gtk::SpinButton theXSpin;
  Gtk::SpinButton theXUpBoundSpin;
  Gtk::SpinButton theYLowBoundSpin;
  Gtk::SpinButton theYSpin;
  Gtk::SpinButton theYUpBoundSpin;
  Gtk::SpinButton theZLowBoundSpin;
  Gtk::SpinButton theZSpin;
  Gtk::SpinButton theZUpBoundSpin;
  Gtk::ToggleButton m_3d;
  Gtk::ToggleButton m_showSurface;
  Gtk::ToggleButton m_showTime;
  Gtk::ToggleButton theRecordButton;
  Gtk::VBox theBoxCtrl;
  Gtk::VBox theBoxInBound;
  Gtk::VBox theBoxInFrame;
  Gtk::VBox theBoxInLattice;
  Gtk::VBox theBoxInScreen;
  Gtk::ToolButton play_button_;
  Gtk::HBox progress_box_;
  Gtk::Adjustment progress_adj_;
  Gtk::SpinButton progress_spin_;
  Gtk::HScale progress_bar_;
};

class GLScene : public Gtk::GL::DrawingArea
{
public:
  Color getColor(unsigned int i) { return theSpeciesColor[i]; };
  Color getSpeciesColor(unsigned int id);
  GLScene(const Glib::RefPtr<const Gdk::GL::Config>& config,
          const char* aFileName);
  bool getSpeciesVisibility(unsigned int id);
  bool writePng();
  char* getSpeciesName(unsigned int id);
  static const unsigned int TIMEOUT_INTERVAL;
  unsigned int getColSize() { return theColSize; };
  unsigned int getLayerSize() { return theLayerSize; };
  unsigned int getRowSize() { return theRowSize; };
  unsigned int getSpeciesSize() { return theTotalSpeciesSize; };
  virtual ~GLScene();
  void drawTime();
  void invalidate() { get_window()->invalidate_rect(get_allocation(), false); }
  void pause();
  void play();
  void renderLayout(Glib::RefPtr<Pango::Layout>);
  void resetBound();
  void resetTime();
  void resetView();
  void rotate(int aMult, int x, int y, int z);
  void rotateMidAxis(int aMult, int x, int y, int z);
  void rotateMidAxisAbs(double, int , int , int );
  void set3DMolecule(bool is3D);
  void setInvertBound(bool);
  void setBackgroundColor(Color);
  void setControlBox(ControlBox* aControl);
  void setRecord(bool isRecord);
  void set_is_forward(bool is_forward);
  void setShowSurface(bool);
  void setShowTime(bool);
  void setSpeciesColor(unsigned int id, Color);
  void setSpeciesVisibility(unsigned int id, bool isVisible);
  void setXLowBound( unsigned int aBound );
  void setXUpBound( unsigned int aBound );
  void setYLowBound( unsigned int aBound );
  void setYUpBound( unsigned int aBound );
  void setZLowBound( unsigned int aBound );
  void setZUpBound( unsigned int aBound );
  void step();
  void translate(int x, int y, int z);
  void update() { get_window()->process_updates(false); }
  void zoomIn();
  void zoomOut();
  void set_frame_cnt(int);
  unsigned get_frame_size();
  int get_frame_cnt();
  bool get_is_playing();
protected: 
  void project();
  void set_position(double x, double y, double& px, double& py, double& pz); 
  bool get_is_event_masked(GdkEventButton* event, int mask);
  bool get_is_button(GdkEventButton* event, int button, int mask);
  void init_frames();
  void inc_dec_frame_cnt();
  bool (GLScene::*theLoadCoordsFunction)(std::streampos&);
  bool loadCoords(std::streampos&);
  bool loadMeanCoords(std::streampos&);
  virtual bool on_configure_event(GdkEventConfigure* event);
  virtual bool on_expose_event(GdkEventExpose* event);
  virtual bool on_map_event(GdkEventAny* event);
  virtual bool on_timeout();
  virtual bool on_unmap_event(GdkEventAny* event);
  virtual bool on_visibility_notify_event(GdkEventVisibility* event);
  virtual void on_realize();
  virtual void on_size_allocate(Gtk::Allocation& allocation);
  virtual bool on_button_press_event(GdkEventButton* button_event);
  virtual bool on_button_release_event(GdkEventButton* release_event);
  virtual bool on_scroll_event(GdkEventScroll* scroll_event);
  virtual bool on_motion_notify_event(GdkEventMotion* motion_event);
  virtual bool on_key_press_event(GdkEventKey* key_event);
  void (GLScene::*thePlot3DFunction)();
  void (GLScene::*thePlotFunction)();
  void drawBox(GLfloat xlo, GLfloat xhi, GLfloat ylo, GLfloat yhi, GLfloat zlo,
               GLfloat zhi);
  void drawScene(double);
  void normalizeAngle(double&);
  void plot3DCubicMolecules();
  void plot3DHCPMolecules();
  void plotCubicPoints();
  void plotGrid();
  void plotHCPPoints();
  void plotMean3DCubicMolecules();
  void plotMean3DHCPMolecules();
  void plotMeanHCPPoints();
  void setColor(unsigned int i, Color *c);
  void setLayerColor(unsigned int i);
  void setRandColor(Color *c);
  void setTranslucentColor(unsigned int i, GLfloat j);
  void timeout_add();
  void timeout_remove();
  void configure();
protected:
  Color* theSpeciesColor;
  ControlBox* m_control_;
  GLfloat Aspect;
  GLfloat FieldOfView;
  GLfloat Near;
  GLfloat ViewSize;
  GLfloat X;
  GLfloat Xtrans;
  GLfloat Y;
  GLfloat Ytrans;
  GLfloat Z;
  GLfloat prevX;
  GLfloat prevY;
  GLfloat prevZ;
  GLfloat theBCCc;
  GLfloat hcpO_;
  GLfloat hcpX_;
  GLfloat hcpZ_;
  GLfloat theRotateAngle;
  GLuint m_FontListBase;
  Glib::RefPtr<Pango::Context> ft2_context;
  Glib::ustring m_FontString;
  Glib::ustring m_timeString;
  Point* theMeanPoints;
  Point** thePoints;
  bool *theSpeciesVisibility;
  bool isChanged;
  bool isShownSurface;
  bool isInvertBound;
  bool is_playing_;
  bool is_forward_;
  bool show3DMolecule;
  bool showSurface;
  bool showTime;
  bool startRecord;
  bool is_mouse_rotate_;
  bool is_mouse_zoom_;
  bool is_mouse_pan_;
  bool is_mouse_rotated_;
  char** theSpeciesNameList;
  double *theRadii;
  double theCurrentTime;
  double theRadius;
  double theResetTime;
  double theVoxelRadius;
  double xAngle;
  double yAngle;
  double zAngle;
  double mouse_drag_pos_x_;
  double mouse_drag_pos_y_;
  double mouse_drag_pos_z_;
  double mouse_x_;
  double mouse_y_;
  double z_near_;
  double z_far_;
  double top_;
  double bottom_;
  double left_;
  double right_;
  const double init_zoom_;
  int m_FontHeight;
  int m_FontWidth;
  int frame_cnt_;
  int theGLIndex;
  sigc::connection m_ConnectionTimeout;
  std::ifstream theFile;
  std::map<unsigned int, Point> theCoordPoints;
  std::size_t font_size;
  std::size_t pixel_extent_height;
  std::size_t pixel_extent_width;
  std::size_t tex_height;
  std::size_t tex_width;
  std::vector<std::streampos> frames_;
  std::vector<unsigned int> thePolySpeciesList;
  unsigned int theColSize;
  unsigned int theCutCol;
  unsigned int theCutLayer;
  unsigned int theCutRow;
  unsigned int theDimension;
  unsigned int theLatticeSpSize;
  unsigned int theLatticeType;
  unsigned int theLayerSize;
  unsigned int theLogMarker;
  unsigned int theMeanCount;
  unsigned int theMeanPointSize;
  unsigned int theOffLatticeSpSize;
  unsigned int theOriCol;
  unsigned int theOriLayer;
  unsigned int theOriRow;
  unsigned int thePngNumber;
  unsigned int thePolymerSize;
  unsigned int theReservedSize;
  unsigned int theRowSize;
  unsigned int theScreenHeight;
  unsigned int theScreenWidth;
  unsigned int theStartCoord;
  unsigned int theThreadSize;
  unsigned int theTotalLatticeSpSize;
  unsigned int theTotalOffLatticeSpSize;
  unsigned int theTotalSpeciesSize;
  unsigned int* theMoleculeSize;
  unsigned int* theOffLatticeMoleculeSize;
  unsigned int* theXLowBound;
  unsigned int* theXUpBound;
  unsigned int* theYLowBound;
  unsigned int* theYUpBound;
  unsigned int* theZLowBound;
  unsigned int* theZUpBound;
  unsigned int** theCoords;
  unsigned int** theFrequency;
  Point min_point_;
  Point max_point_;
  Point mid_point_;
  Point dimensions_;
};

class Rulers : public Gtk::Window
{
public:
  Rulers(const Glib::RefPtr<const Gdk::GL::Config>& config,
         const char* aFileName);
protected:
  //signal handlers:
  //Gtk::DrawingArea m_area;
  virtual bool on_key_press_event(GdkEventKey* event);
  GLScene m_area_;
  Gtk::HPaned m_hbox;
  Gtk::HRuler m_hrule;
  Gtk::Table m_table;
  Gtk::VRuler m_vrule;
  ControlBox m_control_;
  bool isRecord;
  static const int XSIZE = 250, YSIZE = 250;
};

#endif /* __Visualizer_hpp */

