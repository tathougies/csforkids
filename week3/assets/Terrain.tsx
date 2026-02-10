<?xml version="1.0" encoding="UTF-8"?>
<tileset version="1.10" tiledversion="1.11.2" name="Terrain" tilewidth="32" tileheight="32" tilecount="1024" columns="32">
 <editorsettings>
  <export target="tileset.json" format="json"/>
 </editorsettings>
 <image source="terrain_atlas.png" width="1024" height="1024"/>
 <tile id="391">
  <properties>
   <property name="water" type="bool" value="true"/>
  </properties>
 </tile>
 <tile id="394">
  <properties>
   <property name="water" type="bool" value="true"/>
  </properties>
 </tile>
 <tile id="437">
  <animation>
   <frame tileid="437" duration="500"/>
   <frame tileid="469" duration="500"/>
  </animation>
 </tile>
 <tile id="438">
  <animation>
   <frame tileid="438" duration="500"/>
   <frame tileid="470" duration="500"/>
  </animation>
 </tile>
 <tile id="439">
  <animation>
   <frame tileid="439" duration="500"/>
   <frame tileid="471" duration="500"/>
  </animation>
 </tile>
 <tile id="451" probability="0.02">
  <properties>
   <property name="water" type="bool" value="true"/>
  </properties>
 </tile>
 <tile id="452">
  <properties>
   <property name="water" type="bool" value="true"/>
  </properties>
 </tile>
 <tile id="453">
  <properties>
   <property name="water" type="bool" value="true"/>
  </properties>
 </tile>
 <tile id="456">
  <properties>
   <property name="water" type="bool" value="true"/>
  </properties>
 </tile>
 <tile id="469">
  <animation>
   <frame tileid="469" duration="500"/>
   <frame tileid="437" duration="500"/>
  </animation>
 </tile>
 <tile id="470">
  <animation>
   <frame tileid="470" duration="500"/>
   <frame tileid="438" duration="500"/>
  </animation>
 </tile>
 <tile id="471">
  <animation>
   <frame tileid="471" duration="500"/>
   <frame tileid="439" duration="500"/>
  </animation>
 </tile>
 <tile id="565" type="Water" probability="0.05">
  <properties>
   <property name="water" type="bool" value="true"/>
  </properties>
  <animation>
   <frame tileid="565" duration="500"/>
   <frame tileid="364" duration="500"/>
   <frame tileid="367" duration="500"/>
   <frame tileid="367" duration="500"/>
  </animation>
 </tile>
 <tile id="566" type="Water" probability="0.05">
  <properties>
   <property name="water" type="bool" value="true"/>
  </properties>
  <animation>
   <frame tileid="566" duration="500"/>
   <frame tileid="368" duration="500"/>
   <frame tileid="365" duration="500"/>
   <frame tileid="368" duration="500"/>
  </animation>
 </tile>
 <tile id="567" type="Water">
  <properties>
   <property name="water" type="bool" value="true"/>
  </properties>
  <animation>
   <frame tileid="369" duration="500"/>
   <frame tileid="366" duration="500"/>
   <frame tileid="567" duration="500"/>
   <frame tileid="369" duration="500"/>
  </animation>
 </tile>
 <wangsets>
  <wangset name="Grass" type="mixed" tile="-1">
   <wangcolor name="Grass" color="#ff0000" tile="-1" probability="1"/>
   <wangcolor name="Water" color="#00ff00" tile="-1" probability="1"/>
   <wangtile tileid="182" wangid="1,1,1,1,1,1,1,1"/>
   <wangtile tileid="183" wangid="1,1,1,1,1,1,1,1"/>
   <wangtile tileid="272" wangid="1,1,1,2,2,2,1,1"/>
   <wangtile tileid="295" wangid="2,2,1,1,1,2,2,2"/>
   <wangtile tileid="296" wangid="2,2,2,2,1,1,1,2"/>
   <wangtile tileid="303" wangid="1,2,2,2,1,1,1,1"/>
   <wangtile tileid="305" wangid="1,1,1,1,1,2,2,2"/>
   <wangtile tileid="327" wangid="1,1,1,2,2,2,2,2"/>
   <wangtile tileid="328" wangid="1,2,2,2,2,2,1,1"/>
   <wangtile tileid="336" wangid="2,2,1,1,1,1,1,2"/>
   <wangtile tileid="358" wangid="1,1,1,2,1,1,1,1"/>
   <wangtile tileid="359" wangid="1,1,1,2,2,2,1,1"/>
   <wangtile tileid="360" wangid="1,1,1,1,1,2,1,1"/>
   <wangtile tileid="390" wangid="1,2,2,2,1,1,1,1"/>
   <wangtile tileid="391" wangid="2,2,2,2,2,2,2,2"/>
   <wangtile tileid="392" wangid="1,1,1,1,1,2,2,2"/>
   <wangtile tileid="422" wangid="1,2,1,1,1,1,1,1"/>
   <wangtile tileid="423" wangid="2,2,1,1,1,1,1,2"/>
   <wangtile tileid="424" wangid="1,1,1,1,1,1,1,2"/>
   <wangtile tileid="565" wangid="2,2,2,2,2,2,2,2"/>
   <wangtile tileid="566" wangid="2,2,2,2,2,2,2,2"/>
  </wangset>
  <wangset name="Water" type="mixed" tile="-1">
   <wangcolor name="Water" color="#ff0000" tile="-1" probability="1"/>
   <wangtile tileid="391" wangid="1,1,1,1,1,1,1,1"/>
   <wangtile tileid="451" wangid="1,1,1,1,1,1,1,1"/>
   <wangtile tileid="565" wangid="1,1,1,1,1,1,1,1"/>
   <wangtile tileid="566" wangid="1,1,1,1,1,1,1,1"/>
  </wangset>
 </wangsets>
</tileset>
