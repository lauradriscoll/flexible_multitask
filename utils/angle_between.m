function ang = angle_between(a,b)
ang = 180*acos((a'*b)/(norm(a)*norm(b)))/pi;
end