"""Quadrature nodes and weights."""

import numpy as np

konrod_15_weights = np.array(
    [
        2.29353220105292249637320080589695919936e-2,
        6.30920926299785532907006631892042866651e-2,
        1.04790010322250183839876322541518017444e-1,
        1.40653259715525918745189590510237920400e-1,
        1.69004726639267902826583426598550284106e-1,
        1.90350578064785409913256402421013682826e-1,
        2.04432940075298892414161999234649084717e-1,
        2.09482141084727828012999174891714263698e-1,
        2.04432940075298892414161999234649084717e-1,
        1.90350578064785409913256402421013682826e-1,
        1.69004726639267902826583426598550284106e-1,
        1.40653259715525918745189590510237920400e-1,
        1.04790010322250183839876322541518017444e-1,
        6.30920926299785532907006631892042866651e-2,
        2.29353220105292249637320080589695919936e-2,
    ]
)
konrod_15_nodes = np.array(
    [
        -9.91455371120812639206854697526328516642e-1,
        -9.49107912342758524526189684047851262401e-1,
        -8.64864423359769072789712788640926201211e-1,
        -7.41531185599394439863864773280788407074e-1,
        -5.86087235467691130294144838258729598437e-1,
        -4.05845151377397166906606412076961463347e-1,
        -2.07784955007898467600689403773244913480e-1,
        0.00000000000000000000000000000000000000e0,
        2.07784955007898467600689403773244913480e-1,
        4.05845151377397166906606412076961463347e-1,
        5.86087235467691130294144838258729598437e-1,
        7.41531185599394439863864773280788407074e-1,
        8.64864423359769072789712788640926201211e-1,
        9.49107912342758524526189684047851262401e-1,
        9.91455371120812639206854697526328516642e-1,
    ]
)
gauss_7_weights = np.array(
    [
        0.00000000000000000000000000000000000000e0,
        1.29484966168869693270611432679082018329e-1,
        0.00000000000000000000000000000000000000e0,
        2.79705391489276667901467771423779582487e-1,
        0.00000000000000000000000000000000000000e0,
        3.81830050505118944950369775488975133878e-1,
        0.00000000000000000000000000000000000000e0,
        4.17959183673469387755102040816326530612e-1,
        0.00000000000000000000000000000000000000e0,
        3.81830050505118944950369775488975133878e-1,
        0.00000000000000000000000000000000000000e0,
        2.79705391489276667901467771423779582487e-1,
        0.00000000000000000000000000000000000000e0,
        1.29484966168869693270611432679082018329e-1,
        0.00000000000000000000000000000000000000e0,
    ]
)
konrod_21_weights = np.array(
    [
        1.16946388673718742780643960621920483962e-2,
        3.25581623079647274788189724593897606174e-2,
        5.47558965743519960313813002445801763737e-2,
        7.50396748109199527670431409161900093952e-2,
        9.31254545836976055350654650833663443900e-2,
        1.09387158802297641899210590325804960272e-1,
        1.23491976262065851077958109831074159512e-1,
        1.34709217311473325928054001771706832761e-1,
        1.42775938577060080797094273138717060886e-1,
        1.47739104901338491374841515972068045524e-1,
        1.49445554002916905664936468389821203745e-1,
        1.47739104901338491374841515972068045524e-1,
        1.42775938577060080797094273138717060886e-1,
        1.34709217311473325928054001771706832761e-1,
        1.23491976262065851077958109831074159512e-1,
        1.09387158802297641899210590325804960272e-1,
        9.31254545836976055350654650833663443900e-2,
        7.50396748109199527670431409161900093952e-2,
        5.47558965743519960313813002445801763737e-2,
        3.25581623079647274788189724593897606174e-2,
        1.16946388673718742780643960621920483962e-2,
    ]
)
konrod_21_nodes = np.array(
    [
        -9.95657163025808080735527280689002847921e-1,
        -9.73906528517171720077964012084452053428e-1,
        -9.30157491355708226001207180059508346225e-1,
        -8.65063366688984510732096688423493048528e-1,
        -7.80817726586416897063717578345042377163e-1,
        -6.79409568299024406234327365114873575769e-1,
        -5.62757134668604683339000099272694140843e-1,
        -4.33395394129247190799265943165784162200e-1,
        -2.94392862701460198131126603103865566163e-1,
        -1.48874338981631210884826001129719984618e-1,
        0.00000000000000000000000000000000000000e0,
        1.48874338981631210884826001129719984618e-1,
        2.94392862701460198131126603103865566163e-1,
        4.33395394129247190799265943165784162200e-1,
        5.62757134668604683339000099272694140843e-1,
        6.79409568299024406234327365114873575769e-1,
        7.80817726586416897063717578345042377163e-1,
        8.65063366688984510732096688423493048528e-1,
        9.30157491355708226001207180059508346225e-1,
        9.73906528517171720077964012084452053428e-1,
        9.95657163025808080735527280689002847921e-1,
    ]
)
gauss_10_weights = np.array(
    [
        0.00000000000000000000000000000000000000e0,
        6.66713443086881375935688098933317928579e-2,
        0.00000000000000000000000000000000000000e0,
        1.49451349150580593145776339657697332403e-1,
        0.00000000000000000000000000000000000000e0,
        2.19086362515982043995534934228163192459e-1,
        0.00000000000000000000000000000000000000e0,
        2.69266719309996355091226921569469352860e-1,
        0.00000000000000000000000000000000000000e0,
        2.95524224714752870173892994651338329421e-1,
        0.00000000000000000000000000000000000000e0,
        2.95524224714752870173892994651338329421e-1,
        0.00000000000000000000000000000000000000e0,
        2.69266719309996355091226921569469352860e-1,
        0.00000000000000000000000000000000000000e0,
        2.19086362515982043995534934228163192459e-1,
        0.00000000000000000000000000000000000000e0,
        1.49451349150580593145776339657697332403e-1,
        0.00000000000000000000000000000000000000e0,
        6.66713443086881375935688098933317928579e-2,
        0.00000000000000000000000000000000000000e0,
    ]
)
konrod_31_weights = np.array(
    [
        5.37747987292334898779205143012764981831e-3,
        1.50079473293161225383747630758072680946e-2,
        2.54608473267153201868740010196533593973e-2,
        3.53463607913758462220379484783600481226e-2,
        4.45897513247648766082272993732796902233e-2,
        5.34815246909280872653431472394302967716e-2,
        6.20095678006706402851392309608029321904e-2,
        6.98541213187282587095200770991474757860e-2,
        7.68496807577203788944327774826590067221e-2,
        8.30805028231330210382892472861037896016e-2,
        8.85644430562117706472754436937743032123e-2,
        9.31265981708253212254868727473457185619e-2,
        9.66427269836236785051799076275893351367e-2,
        9.91735987217919593323931734846031310596e-2,
        1.00769845523875595044946662617569721916e-1,
        1.01330007014791549017374792767492546771e-1,
        1.00769845523875595044946662617569721916e-1,
        9.91735987217919593323931734846031310596e-2,
        9.66427269836236785051799076275893351367e-2,
        9.31265981708253212254868727473457185619e-2,
        8.85644430562117706472754436937743032123e-2,
        8.30805028231330210382892472861037896016e-2,
        7.68496807577203788944327774826590067221e-2,
        6.98541213187282587095200770991474757860e-2,
        6.20095678006706402851392309608029321904e-2,
        5.34815246909280872653431472394302967716e-2,
        4.45897513247648766082272993732796902233e-2,
        3.53463607913758462220379484783600481226e-2,
        2.54608473267153201868740010196533593973e-2,
        1.50079473293161225383747630758072680946e-2,
        5.37747987292334898779205143012764981831e-3,
    ]
)
konrod_31_nodes = np.array(
    [
        -9.98002298693397060285172840152271209073e-1,
        -9.87992518020485428489565718586612581147e-1,
        -9.67739075679139134257347978784337225283e-1,
        -9.37273392400705904307758947710209471244e-1,
        -8.97264532344081900882509656454495882832e-1,
        -8.48206583410427216200648320774216851366e-1,
        -7.90418501442465932967649294817947346862e-1,
        -7.24417731360170047416186054613938009631e-1,
        -6.50996741297416970533735895313274692547e-1,
        -5.70972172608538847537226737253910641238e-1,
        -4.85081863640239680693655740232350612866e-1,
        -3.94151347077563369897207370981045468363e-1,
        -2.99180007153168812166780024266388962662e-1,
        -2.01194093997434522300628303394596207813e-1,
        -1.01142066918717499027074231447392338787e-1,
        0.00000000000000000000000000000000000000e0,
        1.01142066918717499027074231447392338787e-1,
        2.01194093997434522300628303394596207813e-1,
        2.99180007153168812166780024266388962662e-1,
        3.94151347077563369897207370981045468363e-1,
        4.85081863640239680693655740232350612866e-1,
        5.70972172608538847537226737253910641238e-1,
        6.50996741297416970533735895313274692547e-1,
        7.24417731360170047416186054613938009631e-1,
        7.90418501442465932967649294817947346862e-1,
        8.48206583410427216200648320774216851366e-1,
        8.97264532344081900882509656454495882832e-1,
        9.37273392400705904307758947710209471244e-1,
        9.67739075679139134257347978784337225283e-1,
        9.87992518020485428489565718586612581147e-1,
        9.98002298693397060285172840152271209073e-1,
    ]
)
gauss_15_weights = np.array(
    [
        0.00000000000000000000000000000000000000e0,
        3.07532419961172683546283935772044177217e-2,
        0.00000000000000000000000000000000000000e0,
        7.03660474881081247092674164506673384667e-2,
        0.00000000000000000000000000000000000000e0,
        1.07159220467171935011869546685869303416e-1,
        0.00000000000000000000000000000000000000e0,
        1.39570677926154314447804794511028322521e-1,
        0.00000000000000000000000000000000000000e0,
        1.66269205816993933553200860481208811131e-1,
        0.00000000000000000000000000000000000000e0,
        1.86161000015562211026800561866422824506e-1,
        0.00000000000000000000000000000000000000e0,
        1.98431485327111576456118326443839324819e-1,
        0.00000000000000000000000000000000000000e0,
        2.02578241925561272880620199967519314839e-1,
        0.00000000000000000000000000000000000000e0,
        1.98431485327111576456118326443839324819e-1,
        0.00000000000000000000000000000000000000e0,
        1.86161000015562211026800561866422824506e-1,
        0.00000000000000000000000000000000000000e0,
        1.66269205816993933553200860481208811131e-1,
        0.00000000000000000000000000000000000000e0,
        1.39570677926154314447804794511028322521e-1,
        0.00000000000000000000000000000000000000e0,
        1.07159220467171935011869546685869303416e-1,
        0.00000000000000000000000000000000000000e0,
        7.03660474881081247092674164506673384667e-2,
        0.00000000000000000000000000000000000000e0,
        3.07532419961172683546283935772044177217e-2,
        0.00000000000000000000000000000000000000e0,
    ]
)
konrod_41_weights = np.array(
    [
        3.07358371852053150121829324603098748803e-3,
        8.60026985564294219866178795010234725213e-3,
        1.46261692569712529837879603088683561639e-2,
        2.03883734612665235980102314327547051228e-2,
        2.58821336049511588345050670961531429995e-2,
        3.12873067770327989585431193238007378878e-2,
        3.66001697582007980305572407072110084875e-2,
        4.16688733279736862637883059368947380440e-2,
        4.64348218674976747202318809261075168421e-2,
        5.09445739237286919327076700503449486648e-2,
        5.51951053482859947448323724197773291948e-2,
        5.91114008806395723749672206485942171364e-2,
        6.26532375547811680258701221742549805858e-2,
        6.58345971336184221115635569693979431472e-2,
        6.86486729285216193456234118853678017155e-2,
        7.10544235534440683057903617232101674129e-2,
        7.30306903327866674951894176589131127606e-2,
        7.45828754004991889865814183624875286161e-2,
        7.57044976845566746595427753766165582634e-2,
        7.63778676720807367055028350380610018008e-2,
        7.66007119179996564450499015301017408279e-2,
        7.63778676720807367055028350380610018008e-2,
        7.57044976845566746595427753766165582634e-2,
        7.45828754004991889865814183624875286161e-2,
        7.30306903327866674951894176589131127606e-2,
        7.10544235534440683057903617232101674129e-2,
        6.86486729285216193456234118853678017155e-2,
        6.58345971336184221115635569693979431472e-2,
        6.26532375547811680258701221742549805858e-2,
        5.91114008806395723749672206485942171364e-2,
        5.51951053482859947448323724197773291948e-2,
        5.09445739237286919327076700503449486648e-2,
        4.64348218674976747202318809261075168421e-2,
        4.16688733279736862637883059368947380440e-2,
        3.66001697582007980305572407072110084875e-2,
        3.12873067770327989585431193238007378878e-2,
        2.58821336049511588345050670961531429995e-2,
        2.03883734612665235980102314327547051228e-2,
        1.46261692569712529837879603088683561639e-2,
        8.60026985564294219866178795010234725213e-3,
        3.07358371852053150121829324603098748803e-3,
    ]
)
konrod_41_nodes = np.array(
    [
        -9.98859031588277663838315576545863010000e-1,
        -9.93128599185094924786122388471320278223e-1,
        -9.81507877450250259193342994720216944567e-1,
        -9.63971927277913791267666131197277221912e-1,
        -9.40822633831754753519982722212443380274e-1,
        -9.12234428251325905867752441203298113049e-1,
        -8.78276811252281976077442995113078466711e-1,
        -8.39116971822218823394529061701520685330e-1,
        -7.95041428837551198350638833272787942959e-1,
        -7.46331906460150792614305070355641590311e-1,
        -6.93237656334751384805490711845931533386e-1,
        -6.36053680726515025452836696226285936743e-1,
        -5.75140446819710315342946036586425132814e-1,
        -5.10867001950827098004364050955250998425e-1,
        -4.43593175238725103199992213492640107840e-1,
        -3.73706088715419560672548177024927237396e-1,
        -3.01627868114913004320555356858592260615e-1,
        -2.27785851141645078080496195368574624743e-1,
        -1.52605465240922675505220241022677527912e-1,
        -7.65265211334973337546404093988382110048e-2,
        0.00000000000000000000000000000000000000e0,
        7.65265211334973337546404093988382110048e-2,
        1.52605465240922675505220241022677527912e-1,
        2.27785851141645078080496195368574624743e-1,
        3.01627868114913004320555356858592260615e-1,
        3.73706088715419560672548177024927237396e-1,
        4.43593175238725103199992213492640107840e-1,
        5.10867001950827098004364050955250998425e-1,
        5.75140446819710315342946036586425132814e-1,
        6.36053680726515025452836696226285936743e-1,
        6.93237656334751384805490711845931533386e-1,
        7.46331906460150792614305070355641590311e-1,
        7.95041428837551198350638833272787942959e-1,
        8.39116971822218823394529061701520685330e-1,
        8.78276811252281976077442995113078466711e-1,
        9.12234428251325905867752441203298113049e-1,
        9.40822633831754753519982722212443380274e-1,
        9.63971927277913791267666131197277221912e-1,
        9.81507877450250259193342994720216944567e-1,
        9.93128599185094924786122388471320278223e-1,
        9.98859031588277663838315576545863010000e-1,
    ]
)
gauss_20_weights = np.array(
    [
        0.00000000000000000000000000000000000000e0,
        1.76140071391521183118619623518528163621e-2,
        0.00000000000000000000000000000000000000e0,
        4.06014298003869413310399522749321098791e-2,
        0.00000000000000000000000000000000000000e0,
        6.26720483341090635695065351870416063516e-2,
        0.00000000000000000000000000000000000000e0,
        8.32767415767047487247581432220462061002e-2,
        0.00000000000000000000000000000000000000e0,
        1.01930119817240435036750135480349876167e-1,
        0.00000000000000000000000000000000000000e0,
        1.18194531961518417312377377711382287005e-1,
        0.00000000000000000000000000000000000000e0,
        1.31688638449176626898494499748163134916e-1,
        0.00000000000000000000000000000000000000e0,
        1.42096109318382051329298325067164933035e-1,
        0.00000000000000000000000000000000000000e0,
        1.49172986472603746787828737001969436693e-1,
        0.00000000000000000000000000000000000000e0,
        1.52753387130725850698084331955097593492e-1,
        0.00000000000000000000000000000000000000e0,
        1.52753387130725850698084331955097593492e-1,
        0.00000000000000000000000000000000000000e0,
        1.49172986472603746787828737001969436693e-1,
        0.00000000000000000000000000000000000000e0,
        1.42096109318382051329298325067164933035e-1,
        0.00000000000000000000000000000000000000e0,
        1.31688638449176626898494499748163134916e-1,
        0.00000000000000000000000000000000000000e0,
        1.18194531961518417312377377711382287005e-1,
        0.00000000000000000000000000000000000000e0,
        1.01930119817240435036750135480349876167e-1,
        0.00000000000000000000000000000000000000e0,
        8.32767415767047487247581432220462061002e-2,
        0.00000000000000000000000000000000000000e0,
        6.26720483341090635695065351870416063516e-2,
        0.00000000000000000000000000000000000000e0,
        4.06014298003869413310399522749321098791e-2,
        0.00000000000000000000000000000000000000e0,
        1.76140071391521183118619623518528163621e-2,
        0.00000000000000000000000000000000000000e0,
    ]
)
konrod_51_weights = np.array(
    [
        1.98738389233031592650785188284340988943e-3,
        5.56193213535671375804023690106552207018e-3,
        9.47397338617415160720771052365532387165e-3,
        1.32362291955716748136564058469762380776e-2,
        1.68478177091282982315166675363363158404e-2,
        2.04353711458828354565682922359389736788e-2,
        2.40099456069532162200924891648810813929e-2,
        2.74753175878517378029484555178110786148e-2,
        3.07923001673874888911090202152285856009e-2,
        3.40021302743293378367487952295512032257e-2,
        3.71162714834155435603306253676198759960e-2,
        4.00838255040323820748392844670756464014e-2,
        4.28728450201700494768957924394951611020e-2,
        4.55029130499217889098705847526603930437e-2,
        4.79825371388367139063922557569147549836e-2,
        5.02776790807156719633252594334400844406e-2,
        5.23628858064074758643667121378727148874e-2,
        5.42511298885454901445433704598756068261e-2,
        5.59508112204123173082406863827473468203e-2,
        5.74371163615678328535826939395064719948e-2,
        5.86896800223942079619741758567877641398e-2,
        5.97203403241740599790992919325618538354e-2,
        6.05394553760458629453602675175654271623e-2,
        6.11285097170530483058590304162927119227e-2,
        6.14711898714253166615441319652641775865e-2,
        6.15808180678329350787598242400645531904e-2,
        6.14711898714253166615441319652641775865e-2,
        6.11285097170530483058590304162927119227e-2,
        6.05394553760458629453602675175654271623e-2,
        5.97203403241740599790992919325618538354e-2,
        5.86896800223942079619741758567877641398e-2,
        5.74371163615678328535826939395064719948e-2,
        5.59508112204123173082406863827473468203e-2,
        5.42511298885454901445433704598756068261e-2,
        5.23628858064074758643667121378727148874e-2,
        5.02776790807156719633252594334400844406e-2,
        4.79825371388367139063922557569147549836e-2,
        4.55029130499217889098705847526603930437e-2,
        4.28728450201700494768957924394951611020e-2,
        4.00838255040323820748392844670756464014e-2,
        3.71162714834155435603306253676198759960e-2,
        3.40021302743293378367487952295512032257e-2,
        3.07923001673874888911090202152285856009e-2,
        2.74753175878517378029484555178110786148e-2,
        2.40099456069532162200924891648810813929e-2,
        2.04353711458828354565682922359389736788e-2,
        1.68478177091282982315166675363363158404e-2,
        1.32362291955716748136564058469762380776e-2,
        9.47397338617415160720771052365532387165e-3,
        5.56193213535671375804023690106552207018e-3,
        1.98738389233031592650785188284340988943e-3,
    ]
)
konrod_51_nodes = np.array(
    [
        -9.99262104992609834193457486540340593705e-1,
        -9.95556969790498097908784946893901617258e-1,
        -9.88035794534077247637331014577406227072e-1,
        -9.76663921459517511498315386479594067745e-1,
        -9.61614986425842512418130033660167241692e-1,
        -9.42974571228974339414011169658470531905e-1,
        -9.20747115281701561746346084546330631575e-1,
        -8.94991997878275368851042006782804954175e-1,
        -8.65847065293275595448996969588340088203e-1,
        -8.33442628760834001421021108693569569461e-1,
        -7.97873797998500059410410904994306569409e-1,
        -7.59259263037357630577282865204360976388e-1,
        -7.17766406813084388186654079773297780598e-1,
        -6.73566368473468364485120633247622175883e-1,
        -6.26810099010317412788122681624517881020e-1,
        -5.77662930241222967723689841612654067396e-1,
        -5.26325284334719182599623778158010178037e-1,
        -4.73002731445714960522182115009192041332e-1,
        -4.17885382193037748851814394594572487093e-1,
        -3.61172305809387837735821730127640667422e-1,
        -3.03089538931107830167478909980339329200e-1,
        -2.43866883720988432045190362797451586406e-1,
        -1.83718939421048892015969888759528415785e-1,
        -1.22864692610710396387359818808036805532e-1,
        -6.15444830056850788865463923667966312817e-2,
        0.00000000000000000000000000000000000000e0,
        6.15444830056850788865463923667966312817e-2,
        1.22864692610710396387359818808036805532e-1,
        1.83718939421048892015969888759528415785e-1,
        2.43866883720988432045190362797451586406e-1,
        3.03089538931107830167478909980339329200e-1,
        3.61172305809387837735821730127640667422e-1,
        4.17885382193037748851814394594572487093e-1,
        4.73002731445714960522182115009192041332e-1,
        5.26325284334719182599623778158010178037e-1,
        5.77662930241222967723689841612654067396e-1,
        6.26810099010317412788122681624517881020e-1,
        6.73566368473468364485120633247622175883e-1,
        7.17766406813084388186654079773297780598e-1,
        7.59259263037357630577282865204360976388e-1,
        7.97873797998500059410410904994306569409e-1,
        8.33442628760834001421021108693569569461e-1,
        8.65847065293275595448996969588340088203e-1,
        8.94991997878275368851042006782804954175e-1,
        9.20747115281701561746346084546330631575e-1,
        9.42974571228974339414011169658470531905e-1,
        9.61614986425842512418130033660167241692e-1,
        9.76663921459517511498315386479594067745e-1,
        9.88035794534077247637331014577406227072e-1,
        9.95556969790498097908784946893901617258e-1,
        9.99262104992609834193457486540340593705e-1,
    ]
)
gauss_25_weights = np.array(
    [
        0.00000000000000000000000000000000000000e0,
        1.13937985010262879479029641132347736033e-2,
        0.00000000000000000000000000000000000000e0,
        2.63549866150321372619018152952991449360e-2,
        0.00000000000000000000000000000000000000e0,
        4.09391567013063126556234877116459536608e-2,
        0.00000000000000000000000000000000000000e0,
        5.49046959758351919259368915404733241601e-2,
        0.00000000000000000000000000000000000000e0,
        6.80383338123569172071871856567079685547e-2,
        0.00000000000000000000000000000000000000e0,
        8.01407003350010180132349596691113022902e-2,
        0.00000000000000000000000000000000000000e0,
        9.10282619829636498114972207028916533810e-2,
        0.00000000000000000000000000000000000000e0,
        1.00535949067050644202206890392685826988e-1,
        0.00000000000000000000000000000000000000e0,
        1.08519624474263653116093957050116619340e-1,
        0.00000000000000000000000000000000000000e0,
        1.14858259145711648339325545869555808641e-1,
        0.00000000000000000000000000000000000000e0,
        1.19455763535784772228178126512901047390e-1,
        0.00000000000000000000000000000000000000e0,
        1.22242442990310041688959518945851505835e-1,
        0.00000000000000000000000000000000000000e0,
        1.23176053726715451203902873079050142438e-1,
        0.00000000000000000000000000000000000000e0,
        1.22242442990310041688959518945851505835e-1,
        0.00000000000000000000000000000000000000e0,
        1.19455763535784772228178126512901047390e-1,
        0.00000000000000000000000000000000000000e0,
        1.14858259145711648339325545869555808641e-1,
        0.00000000000000000000000000000000000000e0,
        1.08519624474263653116093957050116619340e-1,
        0.00000000000000000000000000000000000000e0,
        1.00535949067050644202206890392685826988e-1,
        0.00000000000000000000000000000000000000e0,
        9.10282619829636498114972207028916533810e-2,
        0.00000000000000000000000000000000000000e0,
        8.01407003350010180132349596691113022902e-2,
        0.00000000000000000000000000000000000000e0,
        6.80383338123569172071871856567079685547e-2,
        0.00000000000000000000000000000000000000e0,
        5.49046959758351919259368915404733241601e-2,
        0.00000000000000000000000000000000000000e0,
        4.09391567013063126556234877116459536608e-2,
        0.00000000000000000000000000000000000000e0,
        2.63549866150321372619018152952991449360e-2,
        0.00000000000000000000000000000000000000e0,
        1.13937985010262879479029641132347736033e-2,
        0.00000000000000000000000000000000000000e0,
    ]
)
konrod_61_weights = np.array(
    [
        1.38901369867700762455159122675969968105e-3,
        3.89046112709988405126720184451550327852e-3,
        6.63070391593129217331982636975016813363e-3,
        9.27327965951776342844114689202436042127e-3,
        1.18230152534963417422328988532505928963e-2,
        1.43697295070458048124514324435800101958e-2,
        1.69208891890532726275722894203220923686e-2,
        1.94141411939423811734089510501284558514e-2,
        2.18280358216091922971674857383389934015e-2,
        2.41911620780806013656863707252320267604e-2,
        2.65099548823331016106017093350754143665e-2,
        2.87540487650412928439787853543342111447e-2,
        3.09072575623877624728842529430922726353e-2,
        3.29814470574837260318141910168539275106e-2,
        3.49793380280600241374996707314678750972e-2,
        3.68823646518212292239110656171359677370e-2,
        3.86789456247275929503486515322810502509e-2,
        4.03745389515359591119952797524681142161e-2,
        4.19698102151642461471475412859697577901e-2,
        4.34525397013560693168317281170732580746e-2,
        4.48148001331626631923555516167232437574e-2,
        4.60592382710069881162717355593735805947e-2,
        4.71855465692991539452614781810994864829e-2,
        4.81858617570871291407794922983045926058e-2,
        4.90554345550297788875281653672381736059e-2,
        4.97956834270742063578115693799423285392e-2,
        5.04059214027823468408930856535850289022e-2,
        5.08817958987496064922974730498046918534e-2,
        5.12215478492587721706562826049442082511e-2,
        5.14261285374590259338628792157812598296e-2,
        5.14947294294515675583404336470993075327e-2,
        5.14261285374590259338628792157812598296e-2,
        5.12215478492587721706562826049442082511e-2,
        5.08817958987496064922974730498046918534e-2,
        5.04059214027823468408930856535850289022e-2,
        4.97956834270742063578115693799423285392e-2,
        4.90554345550297788875281653672381736059e-2,
        4.81858617570871291407794922983045926058e-2,
        4.71855465692991539452614781810994864829e-2,
        4.60592382710069881162717355593735805947e-2,
        4.48148001331626631923555516167232437574e-2,
        4.34525397013560693168317281170732580746e-2,
        4.19698102151642461471475412859697577901e-2,
        4.03745389515359591119952797524681142161e-2,
        3.86789456247275929503486515322810502509e-2,
        3.68823646518212292239110656171359677370e-2,
        3.49793380280600241374996707314678750972e-2,
        3.29814470574837260318141910168539275106e-2,
        3.09072575623877624728842529430922726353e-2,
        2.87540487650412928439787853543342111447e-2,
        2.65099548823331016106017093350754143665e-2,
        2.41911620780806013656863707252320267604e-2,
        2.18280358216091922971674857383389934015e-2,
        1.94141411939423811734089510501284558514e-2,
        1.69208891890532726275722894203220923686e-2,
        1.43697295070458048124514324435800101958e-2,
        1.18230152534963417422328988532505928963e-2,
        9.27327965951776342844114689202436042127e-3,
        6.63070391593129217331982636975016813363e-3,
        3.89046112709988405126720184451550327852e-3,
        1.38901369867700762455159122675969968105e-3,
    ]
)
konrod_61_nodes = np.array(
    [
        -9.99484410050490637571325895705810819469e-1,
        -9.96893484074649540271630050918695283341e-1,
        -9.91630996870404594858628366109485724851e-1,
        -9.83668123279747209970032581605662801940e-1,
        -9.73116322501126268374693868423706884888e-1,
        -9.60021864968307512216871025581797662930e-1,
        -9.44374444748559979415831324037439121586e-1,
        -9.26200047429274325879324277080474004086e-1,
        -9.05573307699907798546522558925958319569e-1,
        -8.82560535792052681543116462530225590057e-1,
        -8.57205233546061098958658510658943856821e-1,
        -8.29565762382768397442898119732501916439e-1,
        -7.99727835821839083013668942322683240736e-1,
        -7.67777432104826194917977340974503131695e-1,
        -7.33790062453226804726171131369527645669e-1,
        -6.97850494793315796932292388026640068382e-1,
        -6.60061064126626961370053668149270753038e-1,
        -6.20526182989242861140477556431189299207e-1,
        -5.79345235826361691756024932172540495907e-1,
        -5.36624148142019899264169793311072794164e-1,
        -4.92480467861778574993693061207708795644e-1,
        -4.47033769538089176780609900322854000162e-1,
        -4.00401254830394392535476211542660633611e-1,
        -3.52704725530878113471037207089373860654e-1,
        -3.04073202273625077372677107199256553531e-1,
        -2.54636926167889846439805129817805107883e-1,
        -2.04525116682309891438957671002024709524e-1,
        -1.53869913608583546963794672743255920419e-1,
        -1.02806937966737030147096751318000592472e-1,
        -5.14718425553176958330252131667225737491e-2,
        0.00000000000000000000000000000000000000e0,
        5.14718425553176958330252131667225737491e-2,
        1.02806937966737030147096751318000592472e-1,
        1.53869913608583546963794672743255920419e-1,
        2.04525116682309891438957671002024709524e-1,
        2.54636926167889846439805129817805107883e-1,
        3.04073202273625077372677107199256553531e-1,
        3.52704725530878113471037207089373860654e-1,
        4.00401254830394392535476211542660633611e-1,
        4.47033769538089176780609900322854000162e-1,
        4.92480467861778574993693061207708795644e-1,
        5.36624148142019899264169793311072794164e-1,
        5.79345235826361691756024932172540495907e-1,
        6.20526182989242861140477556431189299207e-1,
        6.60061064126626961370053668149270753038e-1,
        6.97850494793315796932292388026640068382e-1,
        7.33790062453226804726171131369527645669e-1,
        7.67777432104826194917977340974503131695e-1,
        7.99727835821839083013668942322683240736e-1,
        8.29565762382768397442898119732501916439e-1,
        8.57205233546061098958658510658943856821e-1,
        8.82560535792052681543116462530225590057e-1,
        9.05573307699907798546522558925958319569e-1,
        9.26200047429274325879324277080474004086e-1,
        9.44374444748559979415831324037439121586e-1,
        9.60021864968307512216871025581797662930e-1,
        9.73116322501126268374693868423706884888e-1,
        9.83668123279747209970032581605662801940e-1,
        9.91630996870404594858628366109485724851e-1,
        9.96893484074649540271630050918695283341e-1,
        9.99484410050490637571325895705810819469e-1,
    ]
)
gauss_30_weights = np.array(
    [
        0.00000000000000000000000000000000000000e0,
        7.96819249616660561546588347467362245048e-3,
        0.00000000000000000000000000000000000000e0,
        1.84664683110909591423021319120472690962e-2,
        0.00000000000000000000000000000000000000e0,
        2.87847078833233693497191796112920436396e-2,
        0.00000000000000000000000000000000000000e0,
        3.87991925696270495968019364463476920332e-2,
        0.00000000000000000000000000000000000000e0,
        4.84026728305940529029381404228075178153e-2,
        0.00000000000000000000000000000000000000e0,
        5.74931562176190664817216894020561287971e-2,
        0.00000000000000000000000000000000000000e0,
        6.59742298821804951281285151159623612374e-2,
        0.00000000000000000000000000000000000000e0,
        7.37559747377052062682438500221907341538e-2,
        0.00000000000000000000000000000000000000e0,
        8.07558952294202153546949384605297308759e-2,
        0.00000000000000000000000000000000000000e0,
        8.68997872010829798023875307151257025768e-2,
        0.00000000000000000000000000000000000000e0,
        9.21225222377861287176327070876187671969e-2,
        0.00000000000000000000000000000000000000e0,
        9.63687371746442596394686263518098650964e-2,
        0.00000000000000000000000000000000000000e0,
        9.95934205867952670627802821035694765299e-2,
        0.00000000000000000000000000000000000000e0,
        1.01762389748405504596428952168554044633e-1,
        0.00000000000000000000000000000000000000e0,
        1.02852652893558840341285636705415043868e-1,
        0.00000000000000000000000000000000000000e0,
        1.02852652893558840341285636705415043868e-1,
        0.00000000000000000000000000000000000000e0,
        1.01762389748405504596428952168554044633e-1,
        0.00000000000000000000000000000000000000e0,
        9.95934205867952670627802821035694765299e-2,
        0.00000000000000000000000000000000000000e0,
        9.63687371746442596394686263518098650964e-2,
        0.00000000000000000000000000000000000000e0,
        9.21225222377861287176327070876187671969e-2,
        0.00000000000000000000000000000000000000e0,
        8.68997872010829798023875307151257025768e-2,
        0.00000000000000000000000000000000000000e0,
        8.07558952294202153546949384605297308759e-2,
        0.00000000000000000000000000000000000000e0,
        7.37559747377052062682438500221907341538e-2,
        0.00000000000000000000000000000000000000e0,
        6.59742298821804951281285151159623612374e-2,
        0.00000000000000000000000000000000000000e0,
        5.74931562176190664817216894020561287971e-2,
        0.00000000000000000000000000000000000000e0,
        4.84026728305940529029381404228075178153e-2,
        0.00000000000000000000000000000000000000e0,
        3.87991925696270495968019364463476920332e-2,
        0.00000000000000000000000000000000000000e0,
        2.87847078833233693497191796112920436396e-2,
        0.00000000000000000000000000000000000000e0,
        1.84664683110909591423021319120472690962e-2,
        0.00000000000000000000000000000000000000e0,
        7.96819249616660561546588347467362245048e-3,
        0.00000000000000000000000000000000000000e0,
    ]
)


quad_weights = {
    15: {
        "konrod_weights": konrod_15_weights,
        "konrod_nodes": konrod_15_nodes,
        "gauss_weights": gauss_7_weights,
    },
    21: {
        "konrod_weights": konrod_21_weights,
        "konrod_nodes": konrod_21_nodes,
        "gauss_weights": gauss_10_weights,
    },
    31: {
        "konrod_weights": konrod_31_weights,
        "konrod_nodes": konrod_31_nodes,
        "gauss_weights": gauss_15_weights,
    },
    41: {
        "konrod_weights": konrod_41_weights,
        "konrod_nodes": konrod_41_nodes,
        "gauss_weights": gauss_20_weights,
    },
    51: {
        "konrod_weights": konrod_51_weights,
        "konrod_nodes": konrod_51_nodes,
        "gauss_weights": gauss_25_weights,
    },
    61: {
        "konrod_weights": konrod_61_weights,
        "konrod_nodes": konrod_61_nodes,
        "gauss_weights": gauss_30_weights,
    },
}
