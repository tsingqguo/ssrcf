function seqs=configSeqs_rl

seqOther={
    % domain specific
%     struct('name','fish','path','D:\tracking\OTB\sequences/Fish/','startFrame',1,'endFrame',476,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','freeman3','path','D:\tracking\OTB\sequences/Freeman3/','startFrame',1,'endFrame',460,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','mountainbike','path','D:\tracking\OTB\sequences/MountainBike/','startFrame',1,'endFrame',228,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','lemming','path','D:\tracking\OTB\sequences/Lemming/','startFrame',1,'endFrame',1336,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','tiger2','path','D:\tracking\OTB\sequences/Tiger2/','startFrame',1,'endFrame',365,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),... %% domain specific end
%     struct('name','deer','path','D:\tracking\OTB\sequences/Deer/','startFrame',1,'endFrame',71,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','skating1','path','D:\tracking\OTB\sequences/Skating1/','startFrame',1,'endFrame',400,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','shaking','path','D:\tracking\OTB\sequences/Shaking/','startFrame',1,'endFrame',365,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','singer1','path','D:\tracking\OTB\sequences/Singer1/','startFrame',1,'endFrame',351,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','david','path','D:\tracking\OTB\sequences/David/','startFrame',300,'endFrame',770,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','david2','path','D:\tracking\OTB\sequences/David2/','startFrame',1,'endFrame',537,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','sylvester','path','D:\tracking\OTB\sequences/Sylvester/','startFrame',1,'endFrame',1345,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),... % lammdb = 0.1
%     struct('name','mhyang','path','D:\tracking\OTB\sequences/Mhyang/','startFrame',1,'endFrame',1490,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','doll','path','D:\tracking\OTB\sequences/Doll/','startFrame',1,'endFrame',3872,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','dudek','path','D:\tracking\OTB\sequences/Dudek/','startFrame',1,'endFrame',1145,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','coke','path','D:\tracking\OTB\sequences/Coke/','startFrame',1,'endFrame',291,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','boy','path','D:\tracking\OTB\sequences/Boy/','startFrame',1,'endFrame',602,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','crossing','path','D:\tracking\OTB\sequences/Crossing/','startFrame',1,'endFrame',120,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','couple','path','D:\tracking\OTB\sequences/Couple/','startFrame',1,'endFrame',140,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','football1','path','D:\tracking\OTB\sequences/Football1/','startFrame',1,'endFrame',74,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','jogging-1','path','D:\tracking\OTB\sequences/Jogging/','startFrame',1,'endFrame',307,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','jogging-2','path','D:\tracking\OTB\sequences/Jogging/','startFrame',1,'endFrame',307,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','girl','path','D:\tracking\OTB\sequences/Girl/','startFrame',1,'endFrame',500,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','walking','path','D:\tracking\OTB\sequences/Walking/','startFrame',1,'endFrame',412,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','freeman4','path','D:\tracking\OTB\sequences/Freeman4/','startFrame',1,'endFrame',283,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','jumping','path','D:\tracking\OTB\sequences/Jumping/','startFrame',1,'endFrame',313,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','dog1','path','D:\tracking\OTB\sequences/Dog1/','startFrame',1,'endFrame',1350,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','suv','path','D:\tracking\OTB\sequences/Suv/','startFrame',1,'endFrame',945,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','faceocc1','path','D:\tracking\OTB\sequences/FaceOcc1/','startFrame',1,'endFrame',892,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','faceocc2','path','D:\tracking\OTB\sequences/FaceOcc2/','startFrame',1,'endFrame',812,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','basketball','path','D:\tracking\OTB\sequences/Basketball/','startFrame',1,'endFrame',725,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','football','path','D:\tracking\OTB\sequences/Football/','startFrame',1,'endFrame',362,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','subway','path','D:\tracking\OTB\sequences/Subway/','startFrame',1,'endFrame',175,'nz',4,'ext','jpg','init_rect', [0 0 0 0]),...
%     struct('name','tiger1','path','D:\tracking\OTB\sequences/Tiger1/','startFrame',1,'endFrame',354,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','soccer','path','D:\tracking\OTB\sequences/Soccer/','startFrame',1,'endFrame',392,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','walking2','path','D:\tracking\OTB\sequences/Walking2/','startFrame',1,'endFrame',500,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','carscale','path','D:\tracking\OTB\sequences/CarScale/','startFrame',1,'endFrame',252,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','ironman','path','D:\tracking\OTB\sequences/Ironman/','startFrame',1,'endFrame',166,'nz',4,'ext','jpg','init_rect',[0,0,0,0]),...
    struct('name','skiing','path','D:\tracking\OTB\sequences/Skiing/','startFrame',1,'endFrame',81,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','matrix','path','D:\tracking\OTB\sequences/Matrix/','startFrame',1,'endFrame',100,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),... % lr = 0.8
%     struct('name','freeman1','path','D:\tracking\OTB\sequences/Freeman1/','startFrame',1,'endFrame',326,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
};

seqBetter = { 
%     struct('name','woman','path','D:\tracking\OTB\sequences/Woman/','startFrame',1,'endFrame',597,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','trellis','path','D:\tracking\OTB\sequences/Trellis/','startFrame',1,'endFrame',569,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','singer2','path','D:\tracking\OTB\sequences/Singer2/','startFrame',1,'endFrame',366,'nz',4,'ext','jpg','init_rect',[0,0,0,0]),... % lamdb =0.0001
%     struct('name','car4','path','D:\tracking\OTB\sequences/Car4/','startFrame',1,'endFrame',659,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','fleetface','path','D:\tracking\OTB\sequences/FleetFace/','startFrame',1,'endFrame',707,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','bolt','path','D:\tracking\OTB\sequences/Bolt/','startFrame',1,'endFrame',350,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','liquor','path','D:\tracking\OTB\sequences/Liquor/','startFrame',1,'endFrame',1741,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','cardark','path','D:\tracking\OTB\sequences/CarDark/','startFrame',1,'endFrame',393,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','david3','path','D:\tracking\OTB\sequences/David3/','startFrame',1,'endFrame',252,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
%     struct('name','motorrolling','path','D:\tracking\OTB\sequences/MotorRolling/','startFrame',1,'endFrame',164,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
};

seqs=[seqOther,seqBetter];